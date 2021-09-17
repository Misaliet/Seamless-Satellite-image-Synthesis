from .base_options import SBaseOptions


class STrainOptions(SBaseOptions):
    def initialize(self, parser):
        SBaseOptions.initialize(self, parser)
        parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom display port')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--update_html_freq', type=int, default=4000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=10000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla | lsgan ｜ wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        # training parameters
        parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy: linear | step | plateau | cosine')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr_decay_iters', type=int, default=100, help='multiply by a gamma every lr_decay_iters iterations')
        # lambda parameters
        parser.add_argument('--lambda_L1', type=float, default=10.0, help='weight for |B-G(A, E(B))|')
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight on D loss. D(G(A, E(B)))')
        parser.add_argument('--lambda_GAN2', type=float, default=1.0, help='weight on D2 loss, D(G(A, random_z))')
        parser.add_argument('--lambda_GAN3', type=float, default=1.0, help='weight on D3 loss, ...')
        parser.add_argument('--lambda_z', type=float, default=0.5, help='weight for ||E(G(random_z)) - random_z||')
        parser.add_argument('--lambda_kl', type=float, default=0.01, help='weight for KL loss')
        parser.add_argument('--use_same_D', action='store_true', help='if two Ds share the weights or not')

        # new parameters 
        parser.add_argument('--mask', action='store_true', help='add extra channel with a mask image or not')
        parser.add_argument('--lambda_ml', type=float, default=10, help='weight for ML loss')

        parser.add_argument('--lambda_vgg', type=float, default=5, help='weight for VGG loss')

        parser.add_argument('--lambda_d5', type=float, default=10, help='weight for d5 model loss')

        parser.add_argument('--lambda_hr', type=float, default=1, help='weight for hr model loss')
        
        parser.add_argument('--lr_2', type=float, default=0.0002, help='initial learning rate for adam')

        parser.add_argument('--lambda_G2D', type=float, default=1.0, help='weight on G2 as Discriminator')

        parser.add_argument('--lambda_t1', type=float, default=1.0, help='weight on G2 as Discriminator')
        parser.add_argument('--lambda_t2', type=float, default=1.0, help='weight on G2 as Discriminator')


        self.isTrain = True
        return parser
