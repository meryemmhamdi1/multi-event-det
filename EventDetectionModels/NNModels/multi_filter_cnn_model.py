class MultiFilterCNNModel(object):

    def __init__(self, max_sequences, word_index, embed_dim, embedding_matrix, filter_sizes,
                 num_filters, dropout, learning_rate, beta_1, beta_2, epsilon, n_classes):

        sequence_input = Input(shape=(max_sequences,), dtype='int32')
        embedding_layer = Embedding(len(word_index)+1, embed_dim, input_length=max_sequences,
                                    weights=[embedding_matrix], trainable=True, mask_zero=False)(sequence_input)

        filter_sizes = filter_sizes.split(',')
        #embedding_layer_ex = K.expand_dims(embedding_layer, )
        #embedding_layer_ex = Reshape()(embedding_layer)
        conv_0 = Conv1D(num_filters, int(filter_sizes[0]), padding='valid', kernel_initializer='normal', activation='relu')(embedding_layer)

        conv_1 = Conv1D(num_filters, int(filter_sizes[1]), padding='valid', kernel_initializer='normal', activation='relu')(embedding_layer)

        conv_2 = Conv1D(num_filters, int(filter_sizes[2]), padding='valid', kernel_initializer='normal', activation='relu')(embedding_layer)

        maxpool_0 = MaxPooling1D(pool_size=max_sequences - int(filter_sizes[0]) + 1, strides=1, padding='valid')(conv_0)
        maxpool_1 = MaxPooling1D(pool_size=max_sequences - int(filter_sizes[1]) + 1, strides=1, padding='valid')(conv_1)
        maxpool_2 = MaxPooling1D(pool_size=max_sequences - int(filter_sizes[2]) + 1, strides=1, padding='valid')(conv_2)


        merged_tensor = merge([maxpool_0, maxpool_1, maxpool_2], mode='concat', concat_axis=1)


        flatten = Flatten()(merged_tensor)

        # average_pooling = AveragePooling2D(pool_size=(sequence_length,1),strides=(1,1),
        #                                    border_mode='valid', dim_ordering='tf')(inputs)
        #
        # reshape = Reshape()(average_pooling)
        #reshape = Reshape((3*num_filters,))(merged_tensor)
        dropout_layer = Dropout(dropout)(flatten)
        softmax_layer = Dense(output_dim=n_classes, activation='softmax')(dropout_layer)

        # this creates a model that includes
        model = Model(inputs=sequence_input, outputs=softmax_layer)
        adam = Adam(lr=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
        #sgd = optimizers.SGD(lr=0.001, decay=1e-5, momentum=0.9, nesterov=True)

        model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=["accuracy"])

        self.model = model

