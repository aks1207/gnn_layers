class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).
    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off
    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])
            
 -----------------------------------------------------------------------------------------------------------------------------           
class GraphConvolution(Layer):
    """Graph convolution layer. (featureless=True and transform=False) is not supported for now."""

    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, transform=True, init=glorot, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.transform = transform

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                if input_dim == output_dim and not self.transform and not featureless:
                    continue
                self.vars['weights_' + str(i)] = init([input_dim, output_dim],
                                                      name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.dropout:
            if self.sparse_inputs:
                x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
            else:
                x = tf.nn.dropout(x, 1 - self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if 'weights_' + str(i) in self.vars:
                if not self.featureless:
                    pre_sup = dot(x, self.vars['weights_' + str(i)], sparse=self.sparse_inputs)
                else:
                    pre_sup = self.vars['weights_' + str(i)]
            else:
                pre_sup = x
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)
---------------------------------------------------------------------------------------------------------------
class aw_layer(Layer):
    def __init__(self,input_dim,output_dim,placeholders,act,init=trunc_normal,**kwargs):
        super(aw_layer,self).__init__(**kwargs)
        self.act=act
        self.support=placeholders['support']
        with tf.variable_scope('aw_layer'):
            self.vars['weights']=init([input_dim,output_dim])
    def _call(self,inputs):
        y=dot(self.support[0],self.vars['weights'],sparse=True)
        return self.act(y)
class ay_layer(Layer):
    def __init__(self,input_dim,output_dim,placeholders,act,init=trunc_normal,**kwargs):
        super(ay_layer,self).__init__(**kwargs)
        self.act=act
        self.support=placeholders['support']
    def _call(self,inputs):
        x=inputs
        y=dot(self.support[0],x,sparse=True)
        return y
class axw_layer(Layer):
    def __init__(self,input_dim,output_dim,placeholders,act,init=trunc_normal,**kwargs):
        super(aw_layer,self).__init__(**kwargs)
        self.act=act
        self.support=placeholders['support']
        with tf.variable_scope('axw_layer'):
            self.vars['weights']=init([input_dim,output_dim])
    def _call(self,inputs):
        x=inputs
        y1=dot(inputs,self.vars['weights'])
        y=dot(self.support[0],y1,sparse=True)
        return self.act(y)
        
class highway(Layer):
    def __init__(self,input_dim,placeholders,init=trunc_normal,**kwargs):
        super(ay_layer,self).__init__(**kwargs)
        self.act=act
        self.support=placeholders['support']
        with tf.variable_scope('highway_layer'):
            self.vars['highway_weights']=init([input_dim,1])
    def _call(self,inputs):
        a=inputs[0]
        b=inputs[1]
        coeff=self.act(dot(a,self.vars['highway_weights']))
        y=tf.math.multiply(coeff,a)+tf.math.multiply(1-coeff,b)
        return y