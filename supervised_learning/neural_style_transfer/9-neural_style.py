#!/usr/bin/env python3
"""
This module performs Neural Style Transfer on two images
"""
import numpy as np
import tensorflow as tf


class NST:
    """Neural Style Transfer class"""

    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Initializes the NST class with style and content images,
        and the respective weights for content and style cost
        """
        tf.compat.v1.enable_eager_execution()

        # Validate inputs
        if not isinstance(style_image, np.ndarray) or \
           style_image.ndim != 3 or style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(content_image, np.ndarray) or \
           content_image.ndim != 3 or content_image.shape[2] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        if (type(alpha) is not float and type(alpha) is not int) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if (type(beta) is not float and type(beta) is not int) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        # Ensure correct dimensions
        style_h, style_w, style_c = style_image.shape
        content_h, content_w, content_c = content_image.shape

        if style_h <= 0 or style_w <= 0 or style_c != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if content_h <= 0 or content_w <= 0 or content_c != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixels values are between 0 and 1
        and its largest side is 512 pixels
        """
        if type(image) is not np.ndarray or len(image.shape) != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")
        h, w, c = image.shape
        if h <= 0 or w <= 0 or c != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")
        if h > w:
            h_new = 512
            w_new = int(w * (512 / h))
        else:
            w_new = 512
            h_new = int(h * (512 / w))

        resized = tf.image.resize(np.expand_dims(image, axis=0),
                                          size=(h_new, w_new),
                                          method="bicubic"
                  )
        rescaled = resized / 255
        rescaled = tf.clip_by_value(rescaled, 0, 1)
        return rescaled

    def load_model(self):
        """
        Loads the model for Neural Style Transfer
        """

        # load vgg model
        vgg_model = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet')

        # MaxPooling2D - AveragePooling 2D
        vgg_model.save('base')
        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        vgg = tf.keras.models.load_model(
            'base', custom_objects=custom_objects)

        style_outputs = [
            vgg.get_layer(name).output for name in self.style_layers]
        content_outputs = [
            vgg.get_layer(self.content_layer).output]
        model_outputs = style_outputs + content_outputs

        model = tf.keras.models.Model(
            vgg.input, model_outputs, name="model")

        model.trainable = False

        self.model = model

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculates the gram matrix of a layer
        """
        if not (isinstance(input_layer, tf.Tensor) or
                isinstance(input_layer, tf.Variable)) or\
                len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        _, h, w, c = input_layer.shape
        F = tf.reshape(input_layer, (h * w, c))
        n = tf.shape(F)[0]
        gram = tf.matmul(F, F, transpose_a=True)
        gram = tf.expand_dims(gram, axis=0)
        return gram / tf.cast(n, tf.float32)

    def generate_features(self):
        """
        Extracts the features used to calculate neural style cost
        """
        preprocessed_style = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255)
        preprocessed_content = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255)

        style_features = self.model(preprocessed_style)[:-1]
        content_feature = self.model(preprocessed_content)[-1]

        gram_style_features = []

        for feature in style_features:
            gram_style_features.append(self.gram_matrix(feature))

        self.gram_style_features = gram_style_features
        self.content_feature = content_feature

    def layer_style_cost(self, style_output, gram_target):
        """
        Calculates the style cost for a single layer
        """
        if not (isinstance(style_output, tf.Tensor) or
                isinstance(style_output, tf.Variable)) or len(
                    style_output.shape) != 4:
            raise TypeError("style_output must be a tensor of rank 4")

        c = style_output.shape[-1]
        if not isinstance(gram_target, (tf.Tensor, tf.Variable)) or \
           len(gram_target.shape) != 3 or gram_target.shape != (1, c, c):
            raise TypeError(
                "gram_target must be a tensor of shape [1, {}, {}]".format(
                    c, c))
        gram_style = self.gram_matrix(style_output)
        gram = tf.reduce_mean(tf.square(gram_style - gram_target))
        return gram

    def style_cost(self, style_outputs):
        """
        Calculates the style cost for generated image
        """
        length = len(self.style_layers)
        if type(style_outputs) is not list or len(style_outputs) != length:
            raise TypeError(
                "style_outputs must be a list with a length of {}".format(
                    length
                )
            )

        weight = 1 / length
        style_cost = 0
        for i in range(length):
            style_cost += weight * self.layer_style_cost(
                style_outputs[i], self.gram_style_features[i]
            )

        return style_cost

    def content_cost(self, content_output):
        """
        Calculates the content cost for generated image
        """
        expected_shape = self.content_feature.shape
        if not isinstance(content_output, (tf.Tensor, tf.Variable)) \
                or content_output.shape != expected_shape:
            raise TypeError(
                "content_output must be a tensor of shape {}".format(
                    expected_shape
                )
            )

        content_feature = self.content_feature

        content_cost = tf.reduce_mean(tf.square(
            content_output - content_feature
        ))

        return content_cost

    def total_cost(self, generated_image):
        """
        Calculates the total cost for generated image
        """
        expected_shape = self.content_image.shape
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)) or \
                generated_image.shape != expected_shape:
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(
                    expected_shape
                )
            )

        J_content = self.content_cost(self.model(generated_image)[-1])
        J_style = self.style_cost(self.model(generated_image)[:-1])
        J = self.alpha * J_content + self.beta * J_style

        return J, J_content, J_style

    def compute_grads(self, generated_image):
        """
        Calculates the gradients for the tf.tensor generated_image
        """
        generated_image = tf.convert_to_tensor(generated_image)
        if not isinstance(generated_image, tf.Tensor) or\
            generated_image.shape != self.content_image.shape:
                raise TypeError(
                    "generated_image must be a tensor of shape"
                    "{}".format(self.content_image.shape)
                )

        with tf.GradientTape() as tape:
            tape.watch(generated_image)

            J_content = self.content_cost(self.model(generated_image)[-1])
            style_outputs = self.model(generated_image)[:-1]
            J_style = self.style_cost(style_outputs)
            J_total = J_content * self.alpha + J_style * self.beta

        gradients = tape.gradient(J_total, generated_image)

        return gradients, J_total, J_content, J_style

    def generate_image(self, iterations=1000, step=None, lr=0.01, beta1=0.9, beta2=0.99):
        '''
            Generates the neural style image
        '''
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        
        if step is not None:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step >= iterations:
                raise ValueError("step must be positive and less than iterations")
        
        if not isinstance(lr, (float, int)):
            raise TypeError("lr must be a number")
        if lr <= 0:
            raise ValueError("lr must be positive")
        
        if not isinstance(beta1, float):
            raise TypeError("beta1 must be a float")
        if not 0 <= beta1 <= 1:
            raise ValueError("beta1 must be in the range [0, 1]")
        
        if not isinstance(beta2, float):
            raise TypeError("beta2 must be a float")
        if not 0 <= beta2 <= 1:
            raise ValueError("beta2 must be in the range [0, 1]")

        generated_image = tf.Variable(self.content_image, dtype=tf.float32)
        
        optimizer = tf.optimizers.Adam(learning_rate=lr, beta_1=beta1, beta_2=beta2)
        
        best_cost = float('inf')
        best_image = None
        
        for i in range(iterations):
            with tf.GradientTape() as tape:
                gradients, J_total, J_content, J_style = self.compute_grads(generated_image)
            
            optimizer.apply_gradients([(gradients, generated_image)])
            
            generated_image.assign(
                tf.clip_by_value(
                    generated_image, clip_value_min=0.0, clip_value_max=1.0
                )
            )
            
            if J_total < best_cost:
                best_cost = J_total
                best_image = generated_image.numpy()
            
            if step is not None and i % step == 0:
                J_total_np = J_total.numpy()
                J_content_np = J_content.numpy()
                J_style_np = J_style.numpy()
                
                print(
                    "Cost at iteration {i}: {J_total_np}, "
                    "content {J_content_np}, "
                    "style {J_style_np}".format(
                        i=i, J_total_np=J_total_np,
                        J_content_np=J_content_np,
                        J_style_np=J_style_np
                    )
                )
        
        return best_image, best_cost
