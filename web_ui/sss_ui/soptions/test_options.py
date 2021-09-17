from .base_options import SBaseOptions


class STestOptions(SBaseOptions):
    def initialize(self, parser):
        SBaseOptions.initialize(self, parser)
        parser.add_argument('--results_dir', type=str, default='../results/', help='saves results here.')
        parser.add_argument('--phase', type=str, default='val', help='train, val, test, etc')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        parser.add_argument('--n_samples', type=int, default=5, help='#samples')
        parser.add_argument('--no_encode', action='store_true', help='do not produce encoded image')
        parser.add_argument('--sync', action='store_true', help='use the same latent code for different input images')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio for the results')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--blend', action='store_true', help='mask blending during test time.')

        parser.add_argument('--whole', action='store_true', help='save whole level-3 images')
        parser.add_argument('--tc', action='store_true', help='testing color guidance')
        parser.add_argument('--level', type=int, default=5, help='specify which level to use')

        self.isTrain = False
        return parser
