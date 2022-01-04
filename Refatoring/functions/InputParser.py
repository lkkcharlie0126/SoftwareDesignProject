import argparse
from functions.str2bool import str2bool
class InputParser:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--epochs', type=int, default=500)
        parser.add_argument('--max_time', type=int, default=10)
        parser.add_argument('--test_steps', type=int, default=10)
        parser.add_argument('--batch_size', type=int, default=20)
        parser.add_argument('--bidirectional', type=str2bool, default=str2bool('False'))
        # parser.add_argument('--lstm_layers', type=int, default=2)
        parser.add_argument('--num_units', type=int, default=128)

        self.parser = parser
    def setting(self):
        pass

class ParserAami(InputParser):
    def setting(self):
        parser = self.parser
        parser.add_argument('--data_dir', type=str, default='data/s2s_mitbih_aami')
        parser.add_argument('--n_oversampling', type=int, default=10000)
        parser.add_argument('--checkpoint_dir', type=str, default='checkpoints-seq2seq')
        parser.add_argument('--ckpt_name', type=str, default='seq2seq_mitbih.ckpt')
        parser.add_argument('--classes', nargs='+', type=chr,
                            default=['F', 'N', 'S', 'V'])
        args = parser.parse_args()
        return args

class ParserDS1DS2(InputParser):
    def setting(self):
        parser = self.parser
        parser.add_argument('--data_dir', type=str, default='data/s2s_mitbih_aami_DS1DS2')
        parser.add_argument('--n_oversampling', type=int, default=6000)
        parser.add_argument('--checkpoint_dir', type=str, default='checkpoints-seq2seq_DS1DS2')
        parser.add_argument('--ckpt_name', type=str, default='seq2seq_mitbih_DS1DS2.ckpt')
        parser.add_argument('--classes', nargs='+', type=chr,
                            default=['F', 'N', 'S', 'V'])
        args = parser.parse_args()
        return args