import pickle
import tensorflow
import tensorlayer


class NN:
    def __init__(self):
        self._load_data()
        self._embedding_size = 1024

        self._encode_seqs = tensorflow.placeholder(tensorflow.int64, [1, None], 'encode_seqs')
        self._decode_seqs = tensorflow.placeholder(tensorflow.int64, [1, None], 'decode_seqs')

        self._net_out, self._net_rnn = self._get_model(self._encode_seqs, self._decode_seqs)
        self._y = tensorflow.nn.softmax(self._net_out.outputs)

        self._session = tensorflow.Session(config=tensorflow.ConfigProto(allow_soft_placement=True,
                                                                         log_device_placement=False))
        tensorlayer.layers.initialize_global_variables(self._session)
        tensorlayer.files.load_and_assign_npz(self._session, 'data/model.npz', self._net_out)

    def _load_data(self):
        with open('data/metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)

            self._idx2word = metadata['idx2word']
            self._word2idx = metadata['word2idx']

            self._vocab_size = len(self._idx2word)

            self._start_id = self._vocab_size
            self._end_id = self._vocab_size + 1

            self._word2idx.update({'start_id': self._start_id})
            self._word2idx.update({'end_id': self._end_id})

            self._idx2word = self._idx2word + ['start_id', 'end_id']

            self._vocab_size = self._vocab_size + 2

    def _get_model(self, encode, decode):
        with tensorflow.variable_scope('model', reuse=False):
            with tensorflow.variable_scope('embedding') as vs:
                net_encode = tensorlayer.layers.EmbeddingInputlayer(
                    inputs=encode,
                    vocabulary_size=self._vocab_size,
                    embedding_size=self._embedding_size,
                    name='embedding_seqs'
                )

                vs.reuse_variables()
                tensorlayer.layers.set_name_reuse(True)

                net_decode = tensorlayer.layers.EmbeddingInputlayer(
                    inputs=decode,
                    vocabulary_size=self._vocab_size,
                    embedding_size=self._embedding_size,
                    name='embedding_seqs'
                )

            net_rnn = tensorlayer.layers.Seq2Seq(
                net_encode,
                net_decode,
                cell_fn=tensorflow.nn.rnn_cell.BasicLSTMCell,
                n_hidden=self._embedding_size,
                initializer=tensorflow.random_uniform_initializer(-0.1, 0.1),
                encode_sequence_length=tensorlayer.layers.retrieve_seq_length_op2(encode),
                decode_sequence_length=tensorlayer.layers.retrieve_seq_length_op2(decode),
                initial_state_encode=None,
                dropout=None,
                n_layer=3,
                return_seq_2d=True,
                name='seq2seq'
            )

            net_out = tensorlayer.layers.DenseLayer(net_rnn, self._vocab_size, name='output')
        return net_out, net_rnn

    def take_answer(self, msg):
        idxs = [self._word2idx.get(word, self._word2idx['unk']) for word in msg.split(' ')]

        state = self._session.run(self._net_rnn.final_state_encode, {
            self._encode_seqs: [idxs]
        })
        o, state = self._session.run([self._y, self._net_rnn.final_state_decode], {
            self._net_rnn.initial_state_decode: state,
            self._decode_seqs: [[self._start_id]]
        })

        word_idx = tensorlayer.nlp.sample_top(o[0], top_k=3)
        word = self._idx2word[word_idx]
        sentence = [word]

        for _ in range(30):
            o, state = self._session.run([self._y, self._net_rnn.final_state_decode], {
                self._net_rnn.initial_state_decode: state,
                self._decode_seqs: [[word_idx]]
            })

            word_idx = tensorlayer.nlp.sample_top(o[0], top_k=2)
            word = self._idx2word[word_idx]

            if word_idx == self._end_id:
                break
            sentence = sentence + [word]
        return ' '.join(sentence)

