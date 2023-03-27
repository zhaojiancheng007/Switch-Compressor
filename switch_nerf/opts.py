import configargparse
import yaml


def get_opts_base():
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('--config_file', is_config_file=True)

    parser.add_argument('--dataset_type', type=str, default='filesystem', choices=['filesystem', 'memory'],
                        help="""specifies whether to hold all images in CPU memory during training, or whether to write randomized
                        batches or pixels/rays to disk""")
    parser.add_argument('--chunk_paths', type=str, nargs='+', default=None,
                        help="""scratch directory to write shuffled batches to when training using the filesystem dataset. 
    Should be set to a non-existent path when first created, and can then be reused by subsequent training runs once all chunks are written""")
    parser.add_argument('--num_chunks', type=int, default=200,
                        help='number of shuffled chunk files to write to disk. Each chunk should be small enough to fit into CPU memory')
    parser.add_argument('--generate_chunk', default=False, action='store_true',
                        help='only generate chunks')
    parser.add_argument('--disk_flush_size', type=int, default=10000000)
    parser.add_argument('--train_every', type=int, default=1,
                        help='if set to larger than 1, subsamples each n training images')

    parser.add_argument('--cluster_mask_path', type=str, default=None,
                        help='directory containing pixel masks for all training images (generated by create_cluster_masks.py)')

    parser.add_argument('--ckpt_path', type=str, default=None, help='path towards serialized model checkpoint')
    parser.add_argument('--container_path', type=str, default=None,
                        help='path towards merged Mega-NeRF model generated by merged_submodules.py')

    parser.add_argument('--near', type=float, default=1, help='ray near bounds')
    parser.add_argument('--far', type=float, default=None,
                        help='ray far bounds. Will be automatically set if not explicitly set')
    parser.add_argument('--ray_altitude_range', nargs='+', type=float, default=None,
                        help='constrains ray sampling to the given altitude')
    parser.add_argument('--coarse_samples', type=int, default=256,
                        help='number of coarse samples')
    parser.add_argument('--fine_samples', type=int, default=512,
                        help='number of additional fine samples')

    parser.add_argument('--train_scale_factor', type=int, default=1,
                        help='downsamples training images if greater than 1')
    parser.add_argument('--val_scale_factor', type=int, default=4,
                        help='downsamples validation images if greater than 1')

    parser.add_argument('--pos_xyz_dim', type=int, default=12,
                        help='frequency encoding dimension applied to xyz position')
    parser.add_argument('--pos_dir_dim', type=int, default=4,
                        help='frequency encoding dimension applied to view direction (set to 0 to disable)')
    parser.add_argument('--layers', type=int, default=8, help='number of layers in MLP')
    parser.add_argument('--skip_layers', type=int, nargs='+', default=[4], help='indices of the skip connections')
    parser.add_argument('--layer_dim', type=int, default=256, help='number of channels in foreground MLP')
    parser.add_argument('--bg_layer_dim', type=int, default=256, help='number of channels in background MLP')
    parser.add_argument('--appearance_dim', type=int, default=48,
                        help='dimension of appearance embedding vector (set to 0 to disable)')
    parser.add_argument('--affine_appearance', default=False, action='store_true',
                        help='set to true to use affine transformation for appearance instead of latent embedding')

    parser.add_argument('--use_cascade', default=False, action='store_true',
                        help='use separate MLPs to query coarse and fine samples')

    parser.add_argument('--train_mega_nerf', type=str, default=None,
                        help='directory train a Mega-NeRF architecture (point this towards the params.pt file generated by create_cluster_masks.py)')
    parser.add_argument('--boundary_margin', type=float, default=1.15,
                        help='overlap factor between different spatial cells')
    parser.add_argument('--all_val', default=False, action='store_true',
                        help='use all pixels for validation images instead of those specified in cluster masks')
    parser.add_argument('--cluster_2d', default=False, action='store_true', help='cluster without altitude dimension')

    parser.add_argument('--sh_deg', type=int, default=None,
                        help='use spherical harmonics (pos_dir_dim should be set to 0)')

    parser.add_argument('--no_center_pixels', dest='center_pixels', default=True, action='store_false',
                        help='do not shift pixels by +0.5 when computing ray directions')
    parser.add_argument('--no_shifted_softplus', dest='shifted_softplus', default=True, action='store_false',
                        help='use ReLU instead of shifted softplus activation')

    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--image_pixel_batch_size', type=int, default=64 * 1024,
                        help='number of pixels to evaluate per split when rendering validation images')
    parser.add_argument('--model_chunk_size', type=int, default=32 * 1024,
                        help='chunk size to split the input to avoid OOM')

    parser.add_argument('--perturb', type=float, default=1.0, help='factor to perturb depth sampling points')
    parser.add_argument('--noise_std', type=float, default=1.0, help='std dev of noise added to regularize sigma')

    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--lr_decay_factor', type=float, default=0.1, help='learning rate decay factor')

    parser.add_argument('--no_bg_nerf', dest='bg_nerf', default=True, action='store_false',
                        help='do not use background MLP')

    parser.add_argument('--ellipse_scale_factor', type=float, default=1.1, help='Factor to scale foreground bounds')
    parser.add_argument('--no_ellipse_bounds', dest='ellipse_bounds', default=True, action='store_false',
                        help='use spherical foreground bounds instead of ellipse')

    parser.add_argument('--train_iterations', type=int, default=500000, help='training iterations')
    parser.add_argument('--val_interval', type=int, default=500001, help='validation interval')
    parser.add_argument('--ckpt_interval', type=int, default=10000, help='checkpoint interval')

    parser.add_argument('--no_resume_ckpt_state', dest='resume_ckpt_state', default=True, action='store_false')

    parser.add_argument('--no_amp', dest='amp', default=True, action='store_false')
    parser.add_argument('--detect_anomalies', default=False, action='store_true')
    parser.add_argument('--random_seed', type=int, default=42)


    # moe related
    parser.add_argument('--use_moe', default=False, action='store_true',
                        help='whether using moe nerf')
    parser.add_argument('--bg_use_moe', default=False, action='store_true',
                        help='whether using moe nerf for background')
    parser.add_argument('--bg_use_cfg', default=False, action='store_true',
                        help='read the bg_nerf from the config file, if bg_use_moe is True, then bg_use_cfg should also be true')
    parser.add_argument("--moe_expert_num", type=int, default=8, 
                        help='number of expert')
    parser.add_argument("--moe_l_aux_wt", type=float, default=1e-2, 
                        help='l_aux_wt of tutel moe')    
    parser.add_argument("--moe_capacity_factor", type=float, default=1.25, 
                        help='capacity_factor of tutel moe')
    
    parser.add_argument('--model', type=yaml.safe_load, 
                        help='detailed definition of nerf network layers')   
    parser.add_argument('--model_bg', type=yaml.safe_load, 
                        help='detailed definition of bg_nerf network layers')    
    parser.add_argument('--no_expert_parallel', default=True, action='store_true',
                        help='do not use expert parallel in moe')    
    parser.add_argument("--use_balance_loss", default=True, action='store_true', 
                        help='use load balance loss in moe')
    parser.add_argument('--no_use_balance_loss', dest='use_balance_loss', default=True, action='store_false', 
                        help='not use load balance loss in moe')
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric logging')
    parser.add_argument('--find_unused_parameters', default=False, action='store_true',
                        help='whether using moe nerf')
    parser.add_argument('--no_find_unused_parameters', dest='find_unused_parameters', default=False, action='store_false')
    parser.add_argument("--moe_use_residual", default=False, action='store_true', 
                        help='use residual moe')
    parser.add_argument("--moe_expert_type", type=str, default='expertmlp', 
                        help='expert type of the moe')
    parser.add_argument("--moe_train_batch", default=False, action='store_true', 
                        help='use batched moe for training')    
    parser.add_argument("--moe_test_batch", default=False, action='store_true', 
                        help='use batched moe for testing')
    parser.add_argument('--nerfmoe_class_name', type=str, default="NeRFMoE",
                        help='the class name of nerf moe model used in get_nerf_moe_inner')


    # slurm
    parser.add_argument("--use_slurm", action='store_true', default=False, 
                        help='when using slurm platform')

    parser.add_argument("--accumulation_steps", type=int, default=1, 
                        help='gradient accumulation steps')
    
    parser.add_argument("--expertmlp2seqexperts", action='store_true', default=False, 
                        help='convert state_dict of expertmlp to state_dict of seqexperts')
               
    parser.add_argument("--batch_prioritized_routing", action='store_true', default=False, 
                        help='use batch_prioritized_routing in moe gate, extract_critical_nobatch currently not support this')
    parser.add_argument("--no_batch_prioritized_routing", dest="batch_prioritized_routing", action='store_false', default=True, 
                        help='not use batch_prioritized_routing in moe gate')


    # gates related
    parser.add_argument("--moe_return_gates", default=False, action='store_true',
                        help='return gates index for each point')
    parser.add_argument("--return_pts", action='store_true', default=False,
                        help='return the sample points out of render function')
    parser.add_argument("--return_pts_rgb", action='store_true', default=False,
                        help='return the color of sample points out of render function')
    parser.add_argument("--return_pts_alpha", action='store_true', default=False,
                        help='return the alpha of sample points out of render function')
    parser.add_argument('--render_test_points_typ', type=str, nargs='+', default=["coarse"], 
                        help='point cloud from the coarse sample or fine sample or both, currently only support coarse')
    parser.add_argument("--render_test_points_sample_skip", type=int, default=1, 
                        help='skip number for point samples of each pixel to reduce the point cloud size')
    parser.add_argument("--render_test_points_image_num", type=int, default=1, 
                        help='image number for color point clouds indicating expert ids')
    parser.add_argument("--return_pts_class_seg", default=False, action='store_true',
                        help='return the colored segmentation of each centroid')
         

    parser.add_argument("--moe_return_gate_logits", default=False, action='store_true',
                        help='return gate logits after wg and before softmax')

    parser.add_argument("--shuffle_chunk", action='store_true', default=False, 
                        help='random shuffle the order of chunks before loading')

    parser.add_argument("--use_moe_external_gate", action='store_true', default=False, 
                        help='''use a small network as a gate in MoE layers.''')
    parser.add_argument("--use_gate_input_norm", action='store_true', default=False, 
                        help='use norm layer for gate input, support layernorm and batchnorm')
    # block nerf
    parser.add_argument("--data_type", type=str, default="mega_nerf",
                        help='mega_nerf or block_nerf')
    parser.add_argument("--block_train_list_path", type=str, default="mega_nerf/datasets/lists/block_nerf_train.txt",
                        help='tfrecord file names for train')
    parser.add_argument("--block_val_list_path", type=str, default="mega_nerf/datasets/lists/block_nerf_val.txt",
                        help='tfrecord file names for val')

    parser.add_argument("--block_image_hash_id_map_path", type=str, default="mega_nerf/datasets/lists/block_nerf_id_map.json",
                        help='map between image hash and imagbe id, used for appearance embeding')
    parser.add_argument("--shuffle_tfrecord", action='store_true', default=True, 
                        help='random shuffle the order of tfrecord before loading')

    parser.add_argument("--amp_use_bfloat16", action='store_true', default=False, 
                        help='use bfloat16 in amp of pytorch to see if still nan')
    parser.add_argument("--gate_noise", type=float, default=-1.0,
                        help='std of gate noise when use load_importance_los')
    parser.add_argument("--use_load_importance_loss", action='store_true', default=False, 
                        help='use load_importance_loss, gate_noise should above zero')
    parser.add_argument("--compute_balance_loss", action='store_true', default=False, 
                        help='compute_balance_loss when use load_importance_loss, for comparison')

    parser.add_argument("--dispatcher_no_score", action='store_true', default=False, 
                        help='do not multiply socre on moe output, only use for testing')
    parser.add_argument("--dispatcher_no_postscore", action='store_true', default=False, 
                        help='multiply gate score before feeded into moe')

    parser.add_argument("--use_sigma_noise", action='store_true', default=False, 
                        help='use noise for sigma')
    parser.add_argument("--sigma_noise_std", type=float, default=1.0,
                        help='std of noise for sigma')

    parser.add_argument("--no_optimizer_schedulers", action='store_true', default=False, 
                        help='diable learning scheduler')

    parser.add_argument("--data_loader_num_workers", type=int, default=1,
                        help='num_workers arg in data loader')
    parser.add_argument("--disable_check_finite", action='store_true', default=False, 
                        help='disable check_finite after forward for efficiency and stable training')

    parser.add_argument("--compute_memory", action='store_true', default=False, 
                        help='log the max memory in each step')
    parser.add_argument("--white_bkgd", action='store_true', default=False,
                        help='set to render synthetic data on a white bkgd')
    
    parser.add_argument("--render_image_fn_name", type=str, default=None,
                        help='the name of render_image function used in evaluation')
    
    # mip-nerf
    parser.add_argument('--use_mip', default=False, action='store_true',
                        help='whether using mip nerf')
    parser.add_argument("--weights_resample_padding", type=float, default=0.01)
    parser.add_argument('--stop_level_grad', default=True, action="store_true",
                        help='do not backprop across coarse and fine levels')
    parser.add_argument("--rgb_padding", type=float, default=0.001)

    parser.add_argument('--training_step_fn', type=str, default=None, 
                        help='the name of training_step function, _training_step if None')
    parser.add_argument("--moe_layer_num", type=int, default=1, 
                        help='the total number of moe layer')

    # for blocknerf testing
    parser.add_argument('--set_timeout', default=False, action='store_true',
                        help='when using mip nerf')

    parser.add_argument('--apply_on_expert_fn_name', type=str, default=None, 
                        help='apply_on_expert_fn_name in MOELayer forward')
    parser.add_argument("--return_sigma", default=False, action='store_true',
                        help='return sigma in rendering results')
    parser.add_argument("--return_alpha", default=False, action='store_true',
                        help='return alpha in rendering results')

    parser.add_argument('--moe_layer_ids', type=str, nargs='+', default=None,
                        help="""the name of moe layers in model config""")
    
    parser.add_argument("--use_random_background_color", default=False, action='store_true',
                        help='''use_random_background_color when rendering''')
                        
    return parser
