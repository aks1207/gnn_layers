#!/usr/bin/env python
# coding: utf-8

# In[2]:


from tensorflow import keras
import time
import tensorflow
import tensorflow as tf
from tensorflow.keras import layers
import math


# # Highwaylayer

# In[ ]:


class highwaylayer(tf.keras.layers.Layer):
    def __init__(self,dim):
        super(highwaylayer,self).__init__()
        self.l=tf.keras.layers.Dense(dim,activation="sigmoid",use_bias=False)
        self.units=dim
    def call(self,inputs):
        l0_output=inputs[0]
        l1_output=inputs[1]
        T=self.l(l0_output)
        return T*l1_output+(1-T)*l0_output


# In[ ]:


class gcnlayer_axw(tf.keras.layers.Layer):
    def __init__(self,dim,A):
        super(gcnlayer_axw,self).__init__()
        self.l=tf.keras.layers.Dense(dim,activation='relu',use_bias=False)
        self.A=A
        self.units=dim
    def call(self,inputs):
        A=self.A
        return self.l(tf.sparse.sparse_dense_matmul(A,inputs))
    def compute_output_shape(self,input_shape):
        new_shape=(input_shape[0],self.units)
        return new_shape


# In[ ]:


class gcnlayer2(tf.keras.layers.Layer):
    def __init__(self,dim,A1,A2):
        super(gcnlayer2,self).__init__()
        #self.l=layers.Dense(dim,activation=tf.keras.layers.LeakyReLU(alpha=0.1),use_bias=False)
        #self.l=tf.keras.layers.Dense(dim,activation='relu',use_bias=False)
        self.A1=A1
        self.A2=A2
        self.units=dim
    def call(self,inputs):
        inputs1=inputs[0]
        inputs2=inputs[1]
        A1=self.A1
        A2=self.A2
        #return self.l(tf.sparse.sparse_dense_matmul(A,inputs))
        return [tf.sparse.sparse_dense_matmul(A1,inputs1),tf.sparse.sparse_dense_matmul(A2,inputs2)]
class gcnlayer_unitinp2(tf.keras.layers.Layer):
    def __init__(self,dim,A1,A2):
        super(gcnlayer_unitinp2,self).__init__()
        #self.l=layers.Dense(dim,activation=tf.keras.layers.LeakyReLU(alpha=0.1),use_bias=False)
        self.l=tf.keras.layers.Dense(dim,activation='relu',use_bias=False)
        self.A1=A1
        self.A2=A2
        self.units=dim
    def call(self,inputs):
        inputs1=inputs[0]
        inputs2=inputs[1]
        A1=self.A1
        A2=self.A2
        return [self.l(A1),self.l(A2)]
        #return tf.sparse.sparse_dense_matmul(A,inputs)
class inputlayer(tf.keras.layers.Layer):
    def __init__(self,shape):
        super(inputlayer,self).__init__()
        self.init_embeds=self.add_weight(shape=shape, initializer="random_normal", trainable=True)
    def call(self,inputs):
        return self.init_embeds*1


# In[ ]:


class HGCN_init(tf.keras.Model):
    def __init__(self,adj,shape,dim):
        super(HGCN_init,self).__init__()
        self.l1=gcnlayer(dim,adj)
        self.l2=gcnlayer(dim,adj)
        self.inp=inputlayer(shape)
    def call(self,inputs):
        y=self.inp(inputs)
        y=self.l1(y)
        y=self.l2(y)
        return y


# # GIN Layer

# In[ ]:


class CoeffLayer(layers.Layer):
    def __init__(self):
        super(CoeffLayer, self).__init__()
        self.kernel = self.add_weight("kernel",shape=[1,1])
    def call(self, inputs):
        return inputs*(self.kernel+1.0)
class gin(tf.keras.layers.Layer):
    def __init__(self,dim,A,num,sparse=False):
        super(gin,self).__init__()
        self.l1=layers.Dense(dim,activation="tanh",kernel_regularizer='l2',bias_regularizer='l2')
        #self.l2=layers.Dense(dim,activation="tanh",kernel_regularizer='l2',bias_regularizer='l2')
        #self.l_name=f'{num}'+'_GIN'
        self.coeff=CoeffLayer()
        self.A=A
        self.units=dim
        self.sparse=sparse
        self.batch_nrml=tensorflow.keras.layers.BatchNormalization(name='batch_norm_gcn'+str(num))
    def call(self,inputs):
        inputs=self.batch_nrml(inputs)
        A=self.A
        return self.l1(dot(A,inputs,self.sparse)+self.coeff(inputs))
class gin_dynamic_Adj(tf.keras.layers.Layer):
    def __init__(self,dim,num,sparse=False):
        super(gin_dynamic_Adj,self).__init__()
        self.l=layers.Dense(dim,activation="tanh",kernel_regularizer='l2',bias_regularizer='l2')
        #self.l_name=f'{num}'+'_GIN'
        self.coeff=CoeffLayer()
        self.units=dim
        self.sparse=sparse
        self.batch_nrml=tensorflow.keras.layers.BatchNormalization(name='batch_norm_gcn'+str(num))
    def call(self,inputs):
        inputs_1=inputs[1]
        inputs_1=self.batch_nrml(inputs_1)
        A=inputs[0]
        return self.l(dot(A,inputs_1,self.sparse)+self.coeff(inputs_1))


# # GIN models

# In[ ]:


class simple_GIN_model_perturb_adj(tensorflow.keras.Model):
    def __init__(self,layer_dims,shape,A_pos,A_neg):
        super(simple_GIN_model_perturb_adj,self).__init__()
        self.A_pos=A_pos
        self.A_neg=A_neg
        self.layers_dim=layer_dims
        self.init_shape=shape
        self.outputs_pos=[]
        self.outputs_neg=[]
    def build(self,input_shape):
        w_init=tf.keras.initializers.TruncatedNormal(stddev=1/math.sqrt(self.A_pos.shape[-1]))
        self.init_embeds = tf.Variable(
            initial_value=w_init(shape=self.init_shape, dtype="float32"),
            trainable=True,)
        self.agg1=gin_dynamic_Adj(self.layers_dim[0],1,True)
        self.agg2=gin_dynamic_Adj(self.layers_dim[1],2,True)
    def call(self,inputs):
        x=self.init_embeds
        y1_pos=self.agg1([self.A_pos,x])
        y1_neg=self.agg1([self.A_neg,x])
        y2_pos=self.agg2([self.A_pos,y1_pos])
        y2_neg=self.agg2([self.A_neg,y1_neg])
        self.outputs_pos=[x,y1_pos,y2_pos]
        self.outputs_neg=[x,y1_neg,y2_neg]
        y_pos=[]
        for i in self.outputs_pos:
            y_pos.append(tf.nn.l2_normalize(i, 1))
        y_neg=[]
        for i in self.outputs_neg:
            y_neg.append(tf.nn.l2_normalize(i, 1))
        y_pos=tf.concat(y_pos,axis=1)
        y_neg=tf.concat(y_neg,axis=1)
        y_pos=tf.nn.l2_normalize(y_pos,1)
        y_neg=tf.nn.l2_normalize(y_neg,1)
        return y_pos,y_neg


# In[ ]:


class simple_GIN_model(tensorflow.keras.Model):
    def __init__(self,A,layer_dims,shape,agg,use_inputs=False):
        super(simple_GIN_model,self).__init__()
        self.A=A
        self.layers_dim=layer_dims
        self.init_shape=shape
        self.outputs=[]
        self.agg=agg
        self.use_inputs=use_inputs
    def build(self,inputs_shape):
        if self.use_inputs==False:
            self.init_embeds = self.add_weight('init_embeddings',
                                               shape=self.init_shape,
                                               dtype='float32',
                                               trainable=True)
        if self.agg=='GCN':
            self.agg1=alinet_gcn(self.layers_dim[0],self.A,1,True)
            self.agg2=alinet_gcn(self.layers_dim[1],self.A,2,True)
        if self.agg=='GIN':
            self.agg1=gin(self.layers_dim[0],self.A,1,True)
            self.agg2=gin(self.layers_dim[1],self.A,2,True)
    def call(self,inputs):
        if self.use_inputs:
            x=inputs
        else:
            x=self.init_embeds
        y1=self.agg1(x)
        y2=self.agg2(y1)
        self.outputs=[x,y1,y2]
        y=[]
        for i in self.outputs:
            y.append(tf.nn.l2_normalize(i, 1))
        y=tf.concat(y,axis=1)
        y=tf.nn.l2_normalize(y,1)
        return y
class simple_sum_GIN_model(tensorflow.keras.Model):
    def __init__(self,A,layer_dims,shape,agg):
        super(simple_sum_GIN_model,self).__init__()
        self.A=A
        self.layers_dim=layer_dims
        self.init_shape=shape
        self.outputs=[]
        self.agg=agg
    def build(self,inputs_shape):
        self.init_embeds = self.add_weight('init_embeddings',
                                           shape=self.init_shape,
                                           dtype='float32',
                                           trainable=True)
        self.agg1=alinet_gcn(self.layers_dim[0],self.A,1,True)
        self.agg2=gin(self.layers_dim[1],self.A,2,True)
    def call(self,inputs):
        x=self.init_embeds
        y1=self.agg1(x)
        y2=self.agg2(y1)
        self.outputs=[x,y1,y2]
        y=[]
        for i in self.outputs:
            y.append(tf.nn.l2_normalize(i, 1))
        y=tf.concat(y,axis=1)
        y=tf.nn.l2_normalize(y,1)
        return y
class simple_GIN_model_sum_agg(tensorflow.keras.Model):
    def __init__(self,A,layer_dims,shape):
        super(simple_GIN_model_sum_agg,self).__init__()
        self.A=A
        self.layers_dim=layer_dims
        self.init_shape=shape
        self.outputs=[]
    def build(self,inputs_shape):
        self.init_embeds = self.add_weight('init_embeddings',
                                           shape=self.init_shape,
                                           dtype='float32',
                                           trainable=True)
        self.agg1=alinet_gcn(self.layers_dim[0],self.A,1,True)
        self.agg2=alinet_gcn(self.layers_dim[1],self.A,2,True)
    def call(self,inputs):
        x=self.init_embeds
        y1=self.agg1(x)
        y2=self.agg2(y1)
        self.outputs=[x,y1,y2]
        y=[]
        for i in self.outputs:
            y.append(tf.nn.l2_normalize(i, 1))
        y=tf.concat(y,axis=1)
        y=tf.nn.l2_normalize(y,1)
        return y
class simple_GIN_model_multi_rel(tensorflow.keras.Model):
    def __init__(self,A,layer_dims,shape,lookup,agg):
        super(simple_GIN_model_multi_rel,self).__init__()
        self.A=A
        self.layers_dim=layer_dims
        self.init_shape=shape
        self.outputs=[]
        self.lookup=lookup
        self.agg=agg
    def build(self,inputs_shape):
        self.init_embeds = self.add_weight('init_embeddings',
                                           shape=self.init_shape,
                                           dtype='float32',
                                           trainable=True)
        if self.agg=='GIN':
            self.agg1=gin(self.layers_dim[0],self.A,1,True)
            self.agg2=gin(self.layers_dim[1],self.A,2,True)
        if self.agg=='GCN':
            self.agg1=alinet_gcn(self.layers_dim[0],self.A,1,True)
            self.agg2=alinet_gcn(self.layers_dim[1],self.A,2,True)   
    def call(self,inputs):
        self.outputs=[]
        self.outputs.append(self.init_embeds)
        x=tf.nn.embedding_lookup(self.init_embeds,self.lookup)
        y1=self.agg1(x)
        self.outputs.append(y1)
        y1=tf.nn.embedding_lookup(y1,self.lookup)
        y2=self.agg2(y1)
        self.outputs.append(y2)
        #print(self.init_embeds.shape)
        #print(y1.shape)
        #print(y2.shape)
        #self.outputs=[self.init_embeds,y1,y2]
        y=[]
        for i in self.outputs:
            y.append(tf.nn.l2_normalize(i, 1))
        y=tf.concat(y,axis=1)
        y=tf.nn.l2_normalize(y,1)
        return y


# # RGIN

# In[ ]:


class CoeffLayer(layers.Layer):
    def __init__(self,mul=False):
        super(CoeffLayer, self).__init__()
        self.mul=mul
    def build(self,input_shape):
        if self.mul:
            self.kernel = self.add_weight("kernel",shape=[input_shape[0],1])
        else:
            self.kernel = self.add_weight("kernel",shape=[1,1])
    def call(self, inputs):
        return inputs*(self.kernel+1.0)
def relation_coeff_adj2(total_kg,rel_tensor,keys,values,A):
    #print(rel_tensor.shape)
    rel_coeffs=tf.nn.embedding_lookup(rel_tensor,values)
    #print(rel_coeffs.shape)
    rel_coeffs=tf.math.pow(rel_coeffs+1,-1)
    #print('inv')
    #print(rel_coeffs.shape)
    #print(A.values.shape)
    tmp=tf.math.multiply(A.values,rel_coeffs)
    #print(tmp.shape)
    return tf.sparse.SparseTensor(A.indices,tmp,dense_shape=A.shape)
class Rgin(tf.keras.layers.Layer):
    def __init__(self,dim,A,num,keys,values,total_kg,relations_num,sparse=False):
        super(Rgin,self).__init__()
        self.l=tensorflow.keras.layers.Dense(dim,kernel_regularizer='l2',bias_regularizer='l2')
        #self.l1=tensorflow.keras.layers.Dense(dim,kernel_regularizer='l2',bias_regularizer='l2')
        #self.l_name=f'{num}'+'_GIN'
        self.coeff=CoeffLayer(True)
        self.total_kg=total_kg
        self.relation_coeffs=self.add_weight('relation_coeffs',shape=(relations_num,),
                                            dtype='float32',
                                            trainable=True)
        self.init_rel_coeffs=tf.identity(self.relation_coeffs)
        self.A=A
        self.keys=keys
        self.values=values
        self.units=dim
        self.sparse=sparse
        self.batch_nrml=tensorflow.keras.layers.BatchNormalization(name='batch_norm_gcn'+str(num))
    def call(self,inputs):
        inputs=self.batch_nrml(inputs)
        A=relation_coeff_adj2(self.total_kg,self.relation_coeffs,self.keys,self.values,self.A)
        #A=tf.math.multiply(self.A,r_A)
        return self.l(dot(A,inputs,self.sparse)+self.coeff(inputs))
class alinet_gcn(tensorflow.keras.layers.Layer):
    def __init__(self,dim,A,num,sparse=False):
        super(alinet_gcn,self).__init__()
        self.A=A
        self.units=dim
        self.sparse=sparse
        self.kernel_initializer=initializers.get('glorot_uniform')
        self.kernel_regularizer=regularizers.get('l2')
        self.bias_initializer=initializers.get('zeros')
        self.bias_regularizer=regularizers.get('l2')
        self.act=activations.get('tanh')
        self.num=num
    def build(self,input_shape):
        self.w=self.add_weight('gcn_kernel_'+str(self.num),
                                            shape=[input_shape[-1], self.units],
                                            initializer=self.kernel_initializer,
                                            regularizer=self.kernel_regularizer,
                                            dtype='float32',
                                            trainable=True)
        self.w_start=tensorflow.identity(self.w)
        self.b = self.add_weight('gcn_bias_'+str(self.num),
                                    shape=[self.units, ],
                                    initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    dtype='float32',
                                    trainable=True)
        self.b_start=tensorflow.identity(self.b)
        self.batch_nrml=tensorflow.keras.layers.BatchNormalization(name='batch_norm_gcn'+str(self.num))
    def call(self,inputs):
        inputs = self.batch_nrml(inputs)
        return self.act(dot(self.A,dot(inputs,self.w),self.sparse)+self.b)
import numpy as np
class simple_RGIN_model(tensorflow.keras.Model):
    def __init__(self,A,layer_dims,shape,agg,keys,values,total_kg,relations_num,use_inp=False,add_self_loop=False):
        super(simple_RGIN_model,self).__init__()
        self.A=A
        self.layers_dim=layer_dims
        self.init_shape=shape
        self.outputs=[]
        self.agg=agg
        self.keys=keys
        self.values=values
        self.total_kg=total_kg
        self.relations_num=relations_num
        self.use_inp=use_inp
        self.add_self_loop=add_self_loop
    def build(self,inputs_shape):
        std = 1.0 / math.sqrt(self.init_shape[1])
        if self.use_inp==False:
            self.init_embeds = self.add_weight('init_embeddings',
                                               shape=self.init_shape,
                                               dtype='float32',
                                               trainable=True)
#             self.init_embeds = tf.nn.l2_normalize(self.add_weight('init_embeddings',
#                                                shape=self.init_shape,
#                                                dtype='float32',
#                                                trainable=True,initializer=tf.keras.initializers.TruncatedNormal(stddev=std)),1)
            #init_value=tf.keras.initializers.TruncatedNormal(stddev=std)
            #init_value=tf.nn.l2_normalize(init_value,1)
            #self.init_embeds =tf.Variable(initial_value=init_value,trainable=True)
            #self.init_embeds =tf.nn.l2_normalize(tf.Variable(initial_value=init_value,trainable=True),1)
#             self.init_embeds=self.add_weight(shape=self.init_shape, initializer=tf.keras.initializers.TruncatedNormal(stddev=std), trainable=True)
            self.start_value=tf.identity(self.init_embeds)
        if self.agg=='RGIN':
            #self.agg1=Rgin(self.layers_dim[0],self.A,1,self.keys,self.values,self.total_kg,self.relations_num,True)
            if self.add_self_loop:
                new_indices=np.concatenate([self.A.indices.numpy(),np.array([[i,i] for i in range(30000)])])
                new_values=np.concatenate([self.A.values.numpy(),np.array([1.0 for i in range(30000)],dtype=np.float32)])
                sum_adj=tf.sparse.SparseTensor(new_indices,new_values,dense_shape=self.A.shape)
            else:
                sum_adj=self.A
            self.agg1=alinet_gcn(self.layers_dim[0],sum_adj,1,True)
            self.agg2=Rgin(self.layers_dim[1],self.A,2,self.keys,self.values,self.total_kg,self.relations_num,True)
    def call(self,inputs):
        #print("ok")
        if self.use_inp:
            x=inputs
        else:
            x=self.init_embeds
        #print(self.agg1.A.dtype)
        #print(x.dtype)
        y1=self.agg1(x)
        y2=self.agg2(y1)
        self.outputs=[x,y1,y2]
        y=[]
        for i in self.outputs:
            y.append(tf.nn.l2_normalize(i, 1))
        y=tf.concat(y,axis=1)
        y=tf.nn.l2_normalize(y,1)
        return y
from package import pgnn_func
class simple_REPS_new_PGNN_model(tensorflow.keras.Model):
    def __init__(self,A,layer_dims,shape,agg,keys,values,total_kg,relations_num,use_inp=False,add_self_loop=False,
                 pgnn=False,dis1=None,dis2=None,freq=1,seeds=None,nx_dis=None,reg=False):
        super(simple_REPS_new_PGNN_model,self).__init__()
        self.A=A
        self.layers_dim=layer_dims
        self.init_shape=shape
        self.outputs=[]
        self.agg=agg
        self.keys=keys
        self.values=values
        self.total_kg=total_kg
        self.relations_num=relations_num
        self.use_inp=use_inp
        self.add_self_loop=add_self_loop
        self.pgnn=pgnn
        self.dis1=dis1
        self.dis2=dis2
        self.cn=0
        self.anchor_sampling_freq=freq
        self.seeds=seeds
        self.dis_mat=None
        self.nx_dis=nx_dis
    def build(self,inputs_shape):
        std = 1.0 / math.sqrt(self.init_shape[1])
        if self.use_inp==False:
            self.init_embeds = self.add_weight('init_embeddings',
                                               shape=self.init_shape,
                                               dtype='float32',
                                               trainable=True)
#             self.init_embeds = tf.nn.l2_normalize(self.add_weight('init_embeddings',
#                                                shape=self.init_shape,
#                                                dtype='float32',
#                                                trainable=True,initializer=tf.keras.initializers.TruncatedNormal(stddev=std)),1)
            #init_value=tf.keras.initializers.TruncatedNormal(stddev=std)
            #init_value=tf.nn.l2_normalize(init_value,1)
            #self.init_embeds =tf.Variable(initial_value=init_value,trainable=True)
            #self.init_embeds =tf.nn.l2_normalize(tf.Variable(initial_value=init_value,trainable=True),1)
#             self.init_embeds=self.add_weight(shape=self.init_shape, initializer=tf.keras.initializers.TruncatedNormal(stddev=std), trainable=True)
            self.start_value=tf.identity(self.init_embeds)
        if self.agg=='RGIN':
            #self.agg1=Rgin(self.layers_dim[0],self.A,1,self.keys,self.values,self.total_kg,self.relations_num,True)
            if self.add_self_loop:
                new_indices=np.concatenate([self.A.indices.numpy(),np.array([[i,i] for i in range(30000)])])
                new_values=np.concatenate([self.A.values.numpy(),np.array([1.0 for i in range(30000)],dtype=np.float32)])
                sum_adj=tf.sparse.SparseTensor(new_indices,new_values,dense_shape=self.A.shape)
            else:
                sum_adj=self.A
            self.agg1=alinet_gcn(self.layers_dim[0],sum_adj,1,True)
            self.agg2=Rgin(self.layers_dim[1],self.A,2,self.keys,self.values,self.total_kg,self.relations_num,True)
    def call(self,inputs):
        #print("ok")
        self.cn=self.cn%self.anchor_sampling_freq
        self.cn+=1
        if self.use_inp:
            x=inputs
        else:
            x=self.init_embeds
        #print(self.agg1.A.dtype)
        #print(x.dtype)
        y1=self.agg1(x)
        y2=self.agg2(y1)
        self.outputs=[x,y1,y2]
        y=[]
        for i in self.outputs:
            y.append(tf.nn.l2_normalize(i, 1))
        y3=tf.concat(y,axis=1)
        #y=tf.nn.l2_normalize(y,1)
        if self.pgnn:
            if self.cn==1:
                """
                dists_max,dists_argmax=pgnn_func.preselect_anchor(self.dis1,self.dis2,self.seeds)
                indices,values=fun_get_indices_n_values(dists_argmax,dists_max)
                self.dis_mat=tf.sparse.SparseTensor(indices,values,dense_shape=(dists_argmax.shape[0],dists_argmax.shape[0]))
                """
                self.g1_anchors,self.g2_anchors=pgnn_func.preselect_anchor(self.dis1,self.dis2,self.seeds)
                self.f_dis=self.nx_dis[:,self.g1_anchors]+self.nx_dis[:,self.g2_anchors]
                self.f_dis=tf.nn.l2_normalize(self.f_dis,1)
                self.coeff_g1=tf.nn.l2_normalize(self.nx_dis[:,self.g1_anchors],1)
                self.coeff_g2=tf.nn.l2_normalize(self.nx_dis[:,self.g2_anchors],1)
            pos_y=tf.matmul(self.coeff_g1,tf.nn.embedding_lookup(y3,self.g1_anchors))+\
            tf.matmul(self.coeff_g2,tf.nn.embedding_lookup(y3,self.g2_anchors))
            pos_y=tf.concat([tf.math.multiply(y3,tf.reduce_sum(self.f_dis,1,keepdims=True)),pos_y],1)
            #return tf.nn.l2_normalize(tf.concat([y3,pos_y]))
            #self.highway_coeff=self.highway(y3)
            finr=tf.nn.l2_normalize(pos_y,1)
            return finr
        else:
            return tf.nn.l2_normalize(y3,axis=1)
class single_SUM_model(tensorflow.keras.Model):
    def __init__(self,A,layer_dims,shape,agg,keys,values,total_kg,relations_num,use_inp=False):
        super(single_SUM_model,self).__init__()
        self.A=A
        self.layers_dim=layer_dims
        self.init_shape=shape
        self.outputs=[]
        self.agg=agg
        self.keys=keys
        self.values=values
        self.total_kg=total_kg
        self.relations_num=relations_num
        self.use_inp=use_inp
    def build(self,inputs_shape):
        if self.use_inp==False:
            self.init_embeds = self.add_weight('init_embeddings',
                                               shape=self.init_shape,
                                               dtype='float32',
                                               trainable=True)
            self.start_value=tf.identity(self.init_embeds)
        #self.agg1=alinet_gcn(self.layers_dim[0],self.A,1,True)
        self.agg1=gin(self.layers_dim[0],self.A,1,True)
    def call(self,inputs):
        #print("ok")
        if self.use_inp:
            x=inputs
        else:
            x=self.init_embeds
        
        y1=self.agg1(x)
        #y2=self.agg2(y1)
#         self.outputs=[x,y1,y2]
#         y=[]
#         for i in self.outputs:
#             y.append(tf.nn.l2_normalize(i, 1))
#         y=tf.concat(y,axis=1)
        self.outputs=[x,y1]
        y=tf.nn.l2_normalize(y1,1)
        return y

# In[ ]:


class HGCNmodel_init_trainable(tf.keras.Model):
    def __init__(self,A,out_dim,agg,shape):
        super(HGCNmodel_init_trainable,self).__init__()
        if agg=='GCN':
            self.agg1=gcnlayer(out_dim,A)
            self.agg2=gcnlayer(out_dim,A)
        if agg=='GIN':
            self.agg1=gin(out_dim,A)
            self.agg2=gin(out_dim,A)
        self.gate1=highwaylayer(out_dim)
        self.gate2=highwaylayer(out_dim)
        self.inp=inputlayer(shape)
        self.A=A        
    def call(self,inp):
        l0=self.agg1(self.inp(2))
        #l0=self.gate1([inputs,l0])
        l1=self.agg2(l0)
        return l1


# In[ ]:


class gcnlayer(tf.keras.layers.Layer):
    def __init__(self,dim,A):
        super(gcnlayer,self).__init__()
        #self.l=layers.Dense(dim,activation=tf.keras.layers.LeakyReLU(alpha=0.1),use_bias=False)
        #self.l=tf.keras.layers.Dense(dim,activation='relu',use_bias=False)
        self.A=A
        self.units=dim
    def call(self,inputs):
        A=self.A
        #return self.l(tf.sparse.sparse_dense_matmul(A,inputs))
        #return tf.sparse.sparse_dense_matmul(A,inputs)
        return tf.linalg.matmul(A,inputs)

class gcnlayer_unitinp(tf.keras.layers.Layer):
    def __init__(self,dim,A):
        super(gcnlayer_unitinp,self).__init__()
        #self.l=layers.Dense(dim,activation=tf.keras.layers.LeakyReLU(alpha=0.1),use_bias=False)
        self.A=A
        self.units=dim
        self.l=tf.keras.layers.Dense(self.units,activation='relu',use_bias=False,
                             kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.00577)
                            )
    def call(self,inputs):
        A=self.A
        print("called")
        return self.l(A)
        #return tf.sparse.sparse_dense_matmul(A,inputs)
class inputlayer(tf.keras.layers.Layer):
    def __init__(self,shape):
        super(inputlayer,self).__init__()
        self.init_embeds=self.add_weight(shape=shape, initializer="random_normal", trainable=True)
    def call(self,inputs):
        return self.init_embeds*1


# # Attention layer

# In[ ]:


class attenlayer(tf.keras.layers.Layer):
    def __init__(self,A):
        super(attenlayer,self).__init__()
        self.A=A
    def build(self,input_shape):
        #print(input_shape)
        self.l=tf.keras.layers.Dense(input_shape[-1][-1],activation='tanh',use_bias=False)
    def call(self,inputs):
        a=inputs[0]
        b=inputs[1]
        c=tf.add(a,b)/2
        context=self.l(c)
        coeff1=tf.expand_dims(tf.sigmoid(tf.reduce_sum(tf.math.multiply(a,context),axis=1)),axis=1)
        coeff2=tf.expand_dims(tf.sigmoid(tf.reduce_sum(tf.math.multiply(b,context),axis=1)),axis=1)
        coeff=tf.concat([coeff1,coeff2],axis=1)+1e-10
        coeff=tf.linalg.normalize(coeff,axis=1)[0]
        return tf.math.multiply(a,tf.expand_dims(coeff[:,0],1))+tf.math.multiply(b,tf.expand_dims(coeff[:,1],1))


# In[ ]:


class attenlayer_concat(tf.keras.layers.Layer):
    def __init__(self,A):
        super(attenlayer_concat,self).__init__()
        self.A=A
    def build(self,input_shape):
        #print(input_shape)
        self.l=tf.keras.layers.Dense(input_shape[-1][-1],activation='tanh',use_bias=False)
    def call(self,inputs):
        a=inputs[0]
        b=inputs[1]
        c=tf.concat([a,b],axis=1)
        context=self.l(c)
        coeff1=tf.expand_dims(tf.sigmoid(tf.reduce_sum(tf.math.multiply(a,context),axis=1)),axis=1)
        coeff2=tf.expand_dims(tf.sigmoid(tf.reduce_sum(tf.math.multiply(b,context),axis=1)),axis=1)
        coeff=tf.concat([coeff1,coeff2],axis=1)+1e-10
        coeff=tf.linalg.normalize(coeff,axis=1)[0]
        return tf.math.multiply(a,tf.expand_dims(coeff[:,0],1))+tf.math.multiply(b,tf.expand_dims(coeff[:,1],1))


# # ax,axw,xw,aw layers

# In[ ]:


def dot(a,b,sparse=False):
    if sparse:
        return tf.sparse.sparse_dense_matmul(a,b)
    else:
        return tf.linalg.matmul(a,b)


# In[ ]:


import math
class ax(tf.keras.layers.Layer):
    def __init__(self,A):
        super(ax,self).__init__()
        self.A=A
    def call(self,inputs):
        A=self.A
        return tf.linalg.matmul(A,inputs)
class aw(tf.keras.layers.Layer):
    def __init__(self,A,dim,sparse=False):
        super(aw,self).__init__()
        self.A=A
        self.units=dim
        self.sparse=sparse
    def build(self,inputs_shape):
        #w_init = tf.random_normal_initializer()
        w_init=tf.keras.initializers.TruncatedNormal(stddev=1/math.sqrt(self.A.shape[0]))
#         self.w = tf.Variable(
#             initial_value=tf.nn.l2_normalize(w_init(shape=(self.A.shape[0], self.units), dtype="float32"),1),
#             trainable=True,)
        self.w = tf.Variable(
            initial_value=w_init(shape=(self.A.shape[0], self.units), dtype="float32"),
            trainable=True,)        
    def call(self,inputs):
        return tf.nn.relu(dot(self.A,self.w,self.sparse))
class xw(tf.keras.layers.Layer):
    def __init__(self,dim):
        super(xw,self).__init__()
        self.units=dim
    def build(self,inputs_shape):
        self.l=tf.keras.layers.Dense(self.units,activation='tanh',use_bias=False)
    def call(self,inputs):
        return self.l(inputs)
class axw(tf.keras.layers.Layer):
    def __init__(self,A,dim,sparse1=False,sparse2=False):
        super(axw,self).__init__()
        self.units=dim
        self.A=A
        self.sparse1=sparse1
        self.sparse2=sparse2
    def build(self,inputs_shape):
        #w_init = tf.random_normal_initializer()
        w_init=tf.keras.initializers.TruncatedNormal(stddev=1/math.sqrt(inputs_shape[-1]))
        self.w = tf.Variable(
            initial_value=w_init(shape=(inputs_shape[-1], self.units), dtype="float32"),
            trainable=True,)
    def call(self,inputs):
        a=dot(inputs,self.w,self.sparse1)
        b=dot(self.A,a,self.sparse2)
        return tf.nn.relu(b)


# # Attention Models

# In[ ]:


class gcnalign_atten_ax_and_axw(tf.keras.Model):
    def __init__(self,A,dim):
        super(gcnalign_atten_ax_and_axw,self).__init__()
        self.l1=attenlayer_ax_and_axw(A)
        self.l2=ax(A)
        self.l3=xw(dim)
        self.A=A        
    def call(self,inputs):
        a=tf.constant(inputs[0])
        b=self.l3(inputs[1])
        y=self.l1([a,b])
        return self.l2(y)
class gcnalign_atten_aw_and_axw(tf.keras.Model):
    def __init__(self,A,dim):
        super(gcnalign_atten_aw_and_axw,self).__init__()
        self.l1=attenlayer(A)
        self.l2=aw(A,dim)
        self.l3=axw(A,dim)
        self.A=A        
    def call(self,inputs):
        a=self.l2(1.0)
        b=self.l3(inputs)
        y=self.l1([a,b])
        return tf.linalg.matmul(self.A,y)
class gcnalign_atten_aw_and_axw_concat(tf.keras.Model):
    def __init__(self,A,dim,sparse1=False,sparse2=False):
        super(gcnalign_atten_aw_and_axw_concat,self).__init__()
        self.l1=attenlayer_concat(A)
        self.l2=aw(A,dim,sparse2)
        self.l3=axw(A,dim,sparse1,sparse2)
        self.sparse1=sparse1
        self.sparse2=sparse2
        #self.l4=tf.keras.layers.Dense()
        self.A=A        
    def call(self,inputs):
        a=self.l2(1.0)
        #print("aw")
        b=self.l3(inputs)
        #print("axw")
        a=tf.linalg.normalize(a,axis=1)[0]
        b=tf.linalg.normalize(b,axis=1)[0]
        y=self.l1([a,b])
        #print("attn")
        return dot(self.A,y,self.sparse2)


# In[ ]:


class HGCNmodel(tf.keras.Model):
    def __init__(self,A,out_dim,agg):
        super(HGCNmodel,self).__init__()
        if agg=='GCN':
            self.agg1=gcnlayer_unitinp(out_dim,A)
            self.agg2=gcnlayer(out_dim,A)
        if agg=='GIN':
            self.agg1=gin(out_dim,A)
            self.agg2=gin(out_dim,A)
        #self.gate1=highwaylayer(out_dim)
        #self.gate2=highwaylayer(out_dim)
        self.A=A        
    def call(self,inputs):
        l0=self.agg1(inputs)
        #l0=self.gate1([inputs,l0])
        l1=self.agg2(l0)
        #return self.gate2([l0,l1])
        return l1


# In[ ]:


class HGCNmodel2(tf.keras.Model):
    def __init__(self,A1,A2,out_dim,agg):
        super(HGCNmodel2,self).__init__()
        if agg=='GCN':
            self.agg1=gcnlayer_unitinp2(out_dim,A1,A2)
            self.agg2=gcnlayer2(out_dim,A1,A2)
        if agg=='GIN':
            self.agg1=gin(out_dim,A)
            self.agg2=gin(out_dim,A)
        #self.gate1=highwaylayer(out_dim)
        #self.gate2=highwaylayer(out_dim)
        self.A1=A1
        self.A2=A2
    def call(self,inputs):
        l0=self.agg1(inputs)
        #l0=self.gate1([inputs,l0])
        l1=self.agg2(l0)
        #return self.gate2([l0,l1])
        return l1


# # Alinet Model

# In[ ]:


from tensorflow.python.keras import activations,initializers,regularizers
import tensorflow

class alinet_atten(tensorflow.keras.layers.Layer):
    def __init__(self,dim,A,sparse=False):
        super(alinet_atten,self).__init__()
        self.A=A
        self.units=dim
        self.sparse=sparse
        self.kernel_initializer=initializers.get('glorot_uniform')
        self.kernel_regularizer=regularizers.get('l2')
        self.activation=activations.get(None)
    def build(self,input_shape):
        self.kernel = self.add_weight('atten_W',shape=(input_shape[-1], self.units),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)
        self.kernel1 = self.add_weight('atten_M1',shape=(input_shape[-1],input_shape[-1]),
                                       initializer=self.kernel_initializer,
                                       regularizer=self.kernel_regularizer,
                                       trainable=True)
        self.kernel2 = self.add_weight('atten_M2',shape=(input_shape[-1],input_shape[-1]),
                                       initializer=self.kernel_initializer,
                                       regularizer=self.kernel_regularizer,
                                       trainable=True)
        self.kernel_start=tensorflow.identity(self.kernel)
        self.kernel1_start=tensorflow.identity(self.kernel1)
        self.kernel2_start=tensorflow.identity(self.kernel2)
        self.batch_nrml=tensorflow.keras.layers.BatchNormalization(name='batch_norm_atten')
    def call(self,inputs):
        #print(inputs.shape)
        inputs = self.batch_nrml(inputs)
        mapped_inputs = tensorflow.matmul(inputs, self.kernel)
        #x_M1,x_M2
        attention_inputs1 = tensorflow.matmul(inputs, self.kernel1)
        attention_inputs2 = tensorflow.matmul(inputs, self.kernel2)
        #print(attention_inputs1.shape)
        #tanh((x_M1)*x)(30k,500)--->(30k,),(x_M2)*x
        con_sa_1 = tensorflow.reduce_sum(tensorflow.multiply(attention_inputs1, inputs), 1, keepdims=True)
        con_sa_2 = tensorflow.reduce_sum(tensorflow.multiply(attention_inputs2, inputs), 1, keepdims=True)
        con_sa_1 = tensorflow.keras.activations.tanh(con_sa_1)
        con_sa_2 = tensorflow.keras.activations.tanh(con_sa_2)
        #c1,c2
        #A*c1+A*c2.T
        #A*c1-----(30k_30k),(30k,1)
        #A*c2-----(30k,30k),(1,30k)
        con_sa_1 = tensorflow.cast(self.A, dtype=tensorflow.float32) * con_sa_1
        con_sa_2 = tensorflow.cast(self.A, dtype=tensorflow.float32) * tensorflow.transpose(con_sa_2, [1, 0])
        weights = tensorflow.sparse.add(con_sa_1, con_sa_2)
        weights = tensorflow.SparseTensor(indices=weights.indices,
                                  values=tensorflow.nn.leaky_relu(weights.values),
                                  dense_shape=weights.dense_shape)
        attention_adj = tensorflow.sparse.softmax(weights)
        attention_adj = tensorflow.sparse.reshape(attention_adj, shape=self.A.shape)
        value = tensorflow.sparse.sparse_dense_matmul(attention_adj, mapped_inputs)
        return self.activation(value)
class alinet_highway(tensorflow.keras.layers.Layer):
    def __init__(self,input_dim,output_dim):
        super(alinet_highway,self).__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.kernel_initializer=initializers.get('glorot_uniform')
        self.kernel_regularizer=regularizers.get('l2')
        self.activation=activations.get('tanh')
    def build(self,inputs_shape):
        self.kernel = self.add_weight('highway_kernel',
                                      shape=[self.input_dim, self.output_dim],
                                      initializer=self.kernel_initializer,
                                      dtype='float32',
                                      trainable=True)
        self.kernel_start=tensorflow.identity(self.kernel)
        self.batch_nrml1=tensorflow.keras.layers.BatchNormalization(name='highway_inp1')
        #self.batch_nrml2=tensorflow.keras.layers.BatchNormalization(name='highway_inp2')
    def call(self,inputs):
        input1 = inputs[0]
        input2 = inputs[1]
        input1 = self.batch_nrml1(input1)
        input2 = self.batch_nrml1(input2)
        gate = tensorflow.matmul(input1, self.kernel)
        gate = tensorflow.keras.activations.tanh(gate)
        gate = tensorflow.keras.activations.relu(gate)
        output = tensorflow.add(tensorflow.multiply(input2, 1 - gate), tensorflow.multiply(input1, gate))
        return self.activation(output)
class alinet_model(tensorflow.keras.Model):
    def __init__(self,A1,A2,layer_dims,shape):
        super(alinet_model,self).__init__()
        self.A1=A1
        self.A2=A2
        self.layers_dims=layer_dims
        self.init_shape=shape
        self.outputs=[]
    def build(self,input_shape):
        self.init_embeds = self.add_weight('init_embeddings',
                                           shape=self.init_shape,
                                           dtype='float32',
                                           trainable=True)
        self.gcn1=alinet_gcn(self.layers_dims[0],self.A1,1,True)
        self.atten=alinet_atten(self.layers_dims[0],self.A2,True)
        self.highway=alinet_highway(self.layers_dims[0],self.layers_dims[0])
        self.gcn2=alinet_gcn(self.layers_dims[1],self.A1,2,True)
    def call(self,inputs):
        x=self.init_embeds
        #print(x.shape)
        #print(x)
        y1=self.gcn1(x)
        y2=self.atten(x)
        y3=self.highway([y2,y1])
        y4=self.gcn2(y3)
        self.outputs=[x,y3,y4]
        y=[]
        for i in [y3,y4,x]:
            y.append(tf.nn.l2_normalize(i, 1))
        y=tf.concat(y,axis=1)
        y=tf.nn.l2_normalize(y,1)
        return y
class alinet_model_withinputs(tensorflow.keras.Model):
    def __init__(self,A1,A2,layer_dims,shape):
        super(alinet_model_withinputs,self).__init__()
        self.A1=A1
        self.A2=A2
        self.layers_dims=layer_dims
        self.init_shape=shape
        self.outputs=[]
    def build(self,input_shape):
        self.gcn1=alinet_gcn(self.layers_dims[0],self.A1,1,True)
        self.atten=alinet_atten(self.layers_dims[0],self.A2,True)
        self.highway=alinet_highway(self.layers_dims[0],self.layers_dims[0])
        self.gcn2=alinet_gcn(self.layers_dims[1],self.A1,2,True)
    def call(self,inputs):
        x=inputs
        #print(x.shape)
        #print(x)
        y1=self.gcn1(x)
        y2=self.atten(x)
        y3=self.highway([y2,y1])
        y4=self.gcn2(y3)
        self.outputs=[x,y3,y4]
        y=[]
        for i in [y3,y4,x]:
            y.append(tf.nn.l2_normalize(i, 1))
        y=tf.concat(y,axis=1)
        y=tf.nn.l2_normalize(y,1)
        return y
class alinet_model_withoutattention(tensorflow.keras.Model):
    def __init__(self,A1,A2,layer_dims,shape):
        super(alinet_model_withoutattention,self).__init__()
        self.A1=A1
        self.A2=A2
        self.layers_dims=layer_dims
        self.init_shape=shape
        self.outputs=[]
    def build(self,input_shape):
        self.init_embeds = self.add_weight('init_embeddings',
                                           shape=self.init_shape,
                                           dtype='float32',
                                           trainable=True)
        self.gcn1=alinet_gcn(self.layers_dims[0],self.A1,1,True)
        #self.atten=alinet_atten(self.layers_dims[0],self.A2,True)
        #self.highway=alinet_highway(self.layers_dims[0],self.layers_dims[0])
        self.gcn2=alinet_gcn(self.layers_dims[1],self.A1,2,True)
    def call(self,inputs):
        x=self.init_embeds
        #print(x.shape)
        #print(x)
        y1=self.gcn1(x)
        #y2=self.atten(x)
        #y3=self.highway([y2,y1])
        y4=self.gcn2(y1)
        self.outputs=[x,y1,y4]
        y=[]
        for i in self.outputs:
            y.append(tf.nn.l2_normalize(i, 1))
        y=tf.concat(y,axis=1)
        y=tf.nn.l2_normalize(y,1)
        return y


# In[ ]:





# # Original GCN Alignmodel, Highway model

# In[ ]:


class gcnalign_highway(tf.keras.Model):
    def __init__(self,A,dim,sparse1=False,sparse2=False):
        super(gcnalign_highway,self).__init__()
        self.A=A
        self.dim=dim
        self.sparse1=sparse1
        self.sparse2=sparse2
    def build(self,input_shape):
        self.l1=aw(self.A,self.dim,self.sparse2)
        self.l2=axw(self.A,self.dim,self.sparse1,self.sparse2)
        self.l3=highwaylayer(1)
    def call(self,inputs):
        a=self.l1(1.0)
        b=self.l2(inputs)
        y=self.l3([b,a])
        return dot(self.A,y,self.sparse2)


# In[ ]:


class original_gcnalign(tf.keras.Model):
    def __init__(self,A,dim,sparse=False):
        super(original_gcnalign,self).__init__()
        self.A=A
        self.dim=dim
        self.sparse=sparse
    def build(self,input_shape):
        self.l1=aw(self.A,self.dim,self.sparse)
    def call(self,inputs):
        y=self.l1(1.0)
        return tf.nn.l2_normalize(dot(self.A,y,self.sparse),1)
#         y2=dot(self.A,y,self.sparse)
#         tmp=[]
#         for i in [y,y2]:
#             tmp.append(tf.nn.l2_normalize(i,1))
#         tmp=tf.concat(tmp,1)
#         tmp=tf.nn.l2_normalize(tmp,1)
#         return tmp
class original_gcnalign_with_inputs(tf.keras.Model):
    def __init__(self,A,dim,sparse=False):
        super(original_gcnalign_with_inputs,self).__init__()
        self.A=A
        self.units=dim
        self.sparse=sparse
    def build(self,input_shape):
        w_init=tf.keras.initializers.TruncatedNormal(stddev=1/math.sqrt(input_shape[-1]))
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_shape[-1], self.units), dtype="float32"),
            trainable=True,) 
    def call(self,inputs):
        y=dot(self.A,dot(inputs,self.w),self.sparse)
        return dot(self.A,y,self.sparse)


# In[ ]:


class two_highway(tf.keras.Model):
    def __init__(self,A,dim,sparse1=False,sparse2=False):
        super(two_highway,self).__init__()
        self.A=A
        self.dim=dim
        self.sparse1=sparse1
        self.sparse2=sparse2
    def build(self,input_shape):
        self.l1=aw(self.A,self.dim,self.sparse2)
        self.l2=axw(self.A,self.dim,self.sparse1,self.sparse2)
        self.l3=highwaylayer(1)
        self.l4=highwaylayer(1)
        self.l5=axw(self.A,self.dim,self.sparse1,self.sparse2)
        self.att=attenlayer_concat(self.A)
    def call(self,inputs):
        highway1_out=self.l3([self.l2(inputs[0]),self.l1(1.0)])
        a=self.l5(inputs[1])
        b=self.l1(1.0)
        highway2_out=self.l4([a,b])
        y=self.att([highway1_out,highway2_out])
        return dot(self.A,y,self.sparse2)


# # Mi maximization base model

# In[ ]:


class KGIM(tf.keras.Model):
    def __init__(self,A1,A2,dim,sparse=False):
        super(KGIM,self).__init__()
        self.A1=A1
        self.A2=A2
        self.units=dim
        self.sparse=sparse
    def build(self,input_shape):
        #self.w_init = tf.random_normal_initializer()
        self.w_init=tf.keras.initializers.TruncatedNormal(stddev=1/math.sqrt(self.A1.shape[0]))
        self.w = tf.Variable(
            initial_value=self.w_init(shape=(self.A1.shape[0], self.units), dtype="float32"),
            trainable=True,)
        self.w_start=tf.identity(self.w)
    def call(self,inputs):
#         pos_y=tf.linalg.matmul(self.A1,self.l(self.A1))
#         neg_y=tf.linalg.matmul(self.A2,self.l(self.A2))
        pos_y=dot(self.A1,tf.nn.relu(dot(self.A1,self.w,self.sparse)),self.sparse)
        neg_y=dot(self.A2,tf.nn.relu(dot(self.A2,self.w,self.sparse)),self.sparse)
        return pos_y,neg_y


# In[ ]:


class DKGIM(tf.keras.Model):
    def __init__(self,A,dim,sparse=False):
        super(DKGIM,self).__init__()
        self.A=A
        self.units=dim
        self.sparse=sparse
    def build(self,input_shape):
        #w_init = tf.random_normal_initializer()
        self.w_init=tf.keras.initializers.TruncatedNormal(stddev=1/math.sqrt(input_shape[-1][-1]))
        self.w = tf.Variable(
            initial_value=self.w_init(shape=(input_shape[-1][-1], self.units), dtype="float32"),
            trainable=True,)
    def call(self,inputs):
#         print("called")
#         print(self.A.shape)
#         print(inputs[0].shape)
#         print(self.w.shape)
        pos_y=dot(self.A,dot(inputs[0],self.w),self.sparse)
        neg_y=dot(self.A,dot(inputs[1],self.w),self.sparse)
        pos_y=dot(self.A,pos_y,self.sparse)
        neg_y=dot(self.A,neg_y,self.sparse)
        return pos_y,neg_y


# # PGNN Base Model

# In[ ]:


# def dot(a,b,sparse=False):
#     if sparse:
#         return tf.sparse.sparse_dense_matmul(a,b)
#     else:
#         return tf.linalg.matmul(a,b)
# class PGNN(tf.keras.Model):
#     def __init__(self,A1,A2,dim,sparse=False):
#         super(PGNN,self).__init__()
#         self.A1=A1
#         self.A2=A2
#         self.units=dim
#         self.sparse=sparse
#     def build(self,input_shape):
#         self.w1_init=tf.keras.initializers.TruncatedNormal(stddev=1/math.sqrt(self.A1.shape[0]))
#         self.w1 = tf.Variable(
#             initial_value=self.w_init(shape=(self.A1.shape[0], self.units), dtype="float32"),
#             trainable=True,)
#         self.w2_init=tf.keras.initializers.TruncatedNormal(stddev=1/math.sqrt(self.A2.shape[0]))
#         self.w2 = tf.Variable(
#             initial_value=self.w_init(shape=(self.A2.shape[0], self.units), dtype="float32"),
#             trainable=True,)
# #         self.PGNN1=#---------------------
# #         self.PGNN2=#---------------------
#         self.high_lay=highwaylayer(1)
#     def call(self,[data1,data2]):
#         a1w1=dot(self.A1,self.w1,sefl.sparse)
#         a2w2=dot(self.A2,self.w2,self.sparse)
#         pgnn1_y=self.PGNN1(a1w1)
#         pgnn2_y=self.PGNN2(a2w2)
#         eng_rep=self.high_lay([pgnn1_y,a1w1])
#         fr_rep=self.high_lay([pgnn2_y,a2w2])
#         eng_rep=dot(self.A1,eng_rep,self.sparse)
#         fr_rep=dot(self.A2,fr_rep,self.sparse)
#         total_entities=self.A1.shape[0]+self.A2.shape[1]
#         ans=tf.zeros((total_entites,self.units),dtype=tf.float32)
#         ans=tf.tensor_scatter_nd_update(ans,np.array(range(0,30000,2))[:,np.newaxis],eng_rep)
#         ans=tf.tensor_scatter_nd_update(ans,np.array(range(1,30000,2))[:,np.newaxis],fr_rep)
#         return ans

