import tensorflow as tf

class MyDenseLayer(tf.keras.layers.Layer):

    def __init__(self, in_dim, out_dim):

        super(MyDenseLayer, self).__init__()

        self.W = self.add_weight([in_dim, out_dim])
        self.b = self.add_weight([1, out_dim])


    def call(self, inputs):

        z = tf.matmult(self.W, inputs) + self.b
        # tf.nn.relu(z)
        return tf.math.sigmoid(z)


if __name__ == '__main__':

    # d_layer = MyDenseLayer(3,2)
    #
    # layer = tf.keras.layers.Dense(units=2)
    #
    # n_hidden = 10
    # tf.keras.Sequential( tf.keras.layers.Dense(n_hidden), tf.keras.layers.Dense(2))
    #
    # tf.reduce_min( tf.nn.softmax_cross_entropy_with_logits(y, y_pred) )
    # tf.reduce_min( tf.square( tf.subtract(y,y_pred) ) )
    # loss = tf.keras.losses.MSE(y, y_pred)
    #
    # # inputs = [2, 3]
    # # outputs =
    #
    # weights = tf.Variable( [tf.random.normal()] )
    # while True:
    #     with tf.GradientTape() as g:
    #         loss = compute_loss(weights)
    #         gradient = g.gradient(loss, weights)
    #     weights = weights - lr * gradient
    #
    # # tf.keras.optimizers.ADAM
    # # tf.keras.optimizers.ADADELTA
    #
    # model = tf.keras.Sequential(tf.keras.layers.Dense(10), tf.keras.layers.Dense(5) , tf.keras.layers.Dense(2))
    #
    # optimizer = tf.keras.optimizers.ADAM()
    #
    # while True:
    #
    #     y_pred = model(x)
    #
    #     with tf.GradientTape() as tape:
    #         loss = compute_loss(y, y_pred)
    #
    #     grads = tape.gradient(loss, model.trainable_variables)
    #     optimizer.apply_gradients(zip(grads,model.trainable_variables))
    #
    #
    #     loss = compute_loss(model)
    #
    #
    #
    # tf.keras.layers.Dropout(p=0.5)





    print("Hello Tensorflow!")