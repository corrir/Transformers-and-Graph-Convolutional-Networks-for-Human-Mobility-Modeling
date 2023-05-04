
import torch
import argparse
import sys

from network import RnnFactory

class Setting:
    
    ''' Defines all settings in a single place using a command line interface.
    '''
    
    def parse(self):
        self.guess_foursquare = any(['4sq' in argv for argv in sys.argv]) # foursquare has different default args.
        self.guess_gowalla = any(['gowalla' in argv for argv in sys.argv])
        self.guess_NYC = any(['NYC' in argv for argv in sys.argv])
        self.guess_TKY = any(['TKY' in argv for argv in sys.argv])
        self.guess_US = any(['US' in argv for argv in sys.argv])
        self.guess_foursquare_rnd = any(['4sq-rnd' in argv for argv in sys.argv])
        self.guess_gowalla_rnd = any(['gowalla-rnd' in argv for argv in sys.argv])
        self.guess_NYC_gender = any(['NYC-gender' in argv for argv in sys.argv])
        self.guess_NYC_social = any(['NYC-social' in argv for argv in sys.argv])
        self.guess_TKY_gender = any(['TKY-gender' in argv for argv in sys.argv])
        self.guess_TKY_social = any(['TKY-social' in argv for argv in sys.argv])
                
        parser = argparse.ArgumentParser()        
        if self.guess_foursquare:
            if self.guess_NYC:
                if self.guess_NYC_gender:
                    self.parse_foursquare_NYC_gender(parser)
                elif self.guess_NYC_social:
                    self.parse_foursquare_NYC_social(parser)
                else:
                    self.parse_foursquare_NYC(parser)
            elif self.guess_TKY:
                if self.guess_TKY_gender:
                    self.parse_foursquare_TKY_gender(parser)
                elif self.guess_TKY_social:
                    self.parse_foursquare_TKY_social(parser)
                else:
                    self.parse_foursquare_TKY(parser)
            elif self.guess_US:
                self.parse_foursquare_US(parser)
            elif self.guess_foursquare_rnd:
                self.parse_foursquare_rnd(parser)
            else:
                self.parse_foursquare(parser)
        elif self.guess_gowalla:
            if self.guess_gowalla_rnd:
                self.parse_gowalla_rnd(parser)
            else:
                self.parse_gowalla(parser)        
        else:
            self.parse_breadcrumbs(parser)
        self.parse_arguments(parser)                
        args = parser.parse_args()
        
        ###### settings ######
        # training
        self.gpu = args.gpu
        self.hidden_dim = args.hidden_dim
        self.weight_decay = args.weight_decay
        self.learning_rate = args.lr
        self.epochs = args.epochs
        self.rnn_factory = RnnFactory(args.rnn)
        self.is_lstm = self.rnn_factory.is_lstm()
        self.lambda_t = args.lambda_t
        self.lambda_s = args.lambda_s
        
        # data management
        self.dataset_file = './data/{}'.format(args.dataset)
        self.max_users = 0 # 0 = use all available users
        self.sequence_length = 20
        self.batch_size = args.batch_size
        self.min_checkins = 100
        
        # evaluation        
        self.validate_epoch = args.validate_epoch
        self.report_user = args.report_user        
     
        ### CUDA Setup ###
        self.device = torch.device('cpu') if args.gpu == -1 else torch.device('cuda', args.gpu)        
    
    def parse_arguments(self, parser):        
        # training
        parser.add_argument('--gpu', default=-1, type=int, help='the gpu to use')        
        parser.add_argument('--hidden-dim', default=10, type=int, help='hidden dimensions to use')
        parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay regularization')
        parser.add_argument('--lr', default = 0.01, type=float, help='learning rate')
        parser.add_argument('--epochs', default=100, type=int, help='amount of epochs')
        parser.add_argument('--rnn', default='rnn', type=str, help='the GRU implementation to use: [rnn|gru|lstm]')        
        
        # data management
        parser.add_argument('--dataset', default='checkins-gowalla.txt', type=str, help='the dataset under ./data/<dataset.txt> to load')        
        
        # evaluation        
        parser.add_argument('--validate-epoch', default=5, type=int, help='run each validation after this amount of epochs')
        parser.add_argument('--report-user', default=-1, type=int, help='report every x user on evaluation (-1: ignore)')        
    
    def parse_breadcrumbs(self, parser):
        # defaults for gowalla dataset
        print('breadcrumbs')
        parser.add_argument('--batch-size', default=2, type=int, help='amount of users to process in one pass (batching)')
        parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
        parser.add_argument('--lambda_s', default=100, type=float, help='decay factor for spatial data')

    def parse_gowalla(self, parser):
        # defaults for gowalla dataset
        print('gowalla')
        parser.add_argument('--batch-size', default=600, type=int, help='amount of users to process in one pass (batching)')
        parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
        parser.add_argument('--lambda_s', default=100, type=float, help='decay factor for spatial data')
    
    def parse_gowalla_rnd(self, parser):
        # defaults for gowalla dataset
        print('gowalla_rnd')
        parser.add_argument('--batch-size', default=2, type=int, help='amount of users to process in one pass (batching)')
        parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
        parser.add_argument('--lambda_s', default=100, type=float, help='decay factor for spatial data')

    def parse_foursquare(self, parser):
        # defaults for foursquare dataset
        print('4sq')
        parser.add_argument('--batch-size', default=1300, type=int, help='amount of users to process in one pass (batching)')
        parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
        parser.add_argument('--lambda_s', default=100, type=float, help='decay factor for spatial data')
    
    def parse_foursquare_NYC(self, parser):
        # defaults for foursquare dataset
        print('4sq_NYC')
        parser.add_argument('--batch-size', default=500, type=int, help='amount of users to process in one pass (batching)')
        parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
        parser.add_argument('--lambda_s', default=100, type=float, help='decay factor for spatial data')

    def parse_foursquare_TKY(self, parser):
        # defaults for foursquare dataset
        print('4sq_TKY')
        parser.add_argument('--batch-size', default=150, type=int, help='amount of users to process in one pass (batching)')
        parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
        parser.add_argument('--lambda_s', default=100, type=float, help='decay factor for spatial data')

    def parse_foursquare_US(self, parser):
        # defaults for foursquare dataset
        print('4sq_US')
        parser.add_argument('--batch-size', default=78, type=int, help='amount of users to process in one pass (batching)')
        parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
        parser.add_argument('--lambda_s', default=100, type=float, help='decay factor for spatial data')

    def parse_foursquare_NYC_gender(self, parser):
        # defaults for foursquare dataset
        print('4sq_NYC_gender')
        parser.add_argument('--batch-size', default=30, type=int, help='amount of users to process in one pass (batching)')
        parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
        parser.add_argument('--lambda_s', default=100, type=float, help='decay factor for spatial data')

    def parse_foursquare_TKY_gender(self, parser):
        # defaults for foursquare dataset
        print('4sq_TKY_gender')
        parser.add_argument('--batch-size', default=10, type=int, help='amount of users to process in one pass (batching)')
        parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
        parser.add_argument('--lambda_s', default=100, type=float, help='decay factor for spatial data')

    def parse_foursquare_TKY_social(self, parser):
        # defaults for foursquare dataset
        print('4sq_TKY_social')
        parser.add_argument('--batch-size', default=50, type=int, help='amount of users to process in one pass (batching)')
        parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
        parser.add_argument('--lambda_s', default=100, type=float, help='decay factor for spatial data')

    def parse_foursquare_NYC_social(self, parser):
        # defaults for foursquare dataset
        print('4sq_NYC_social')
        parser.add_argument('--batch-size', default=14, type=int, help='amount of users to process in one pass (batching)')
        parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
        parser.add_argument('--lambda_s', default=100, type=float, help='decay factor for spatial data')

    def parse_foursquare_rnd(self, parser):
        # defaults for foursquare dataset
        print('4sq_rnd')
        parser.add_argument('--batch-size', default=2, type=int, help='amount of users to process in one pass (batching)')
        parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
        parser.add_argument('--lambda_s', default=100, type=float, help='decay factor for spatial data')

    def __str__(self):        
        return ('parse with foursquare default settings' if self.guess_foursquare else 'parse with gowalla default settings') + '\n'\
            + 'use device: {}'.format(self.device)


        
