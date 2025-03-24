"""
configuration function:
"""
import yaml


class Config:
    def __init__(self, args):
        """
        read config file
        """
        config_dir = args.config_file
        stage_choices = ['pretrain','validate','inversion','advtrain', 'noisytrain', 'antitrain', 'onlyX', 'Gan', 'train_shadow', 'queryfree', 'topaz', 'onlyM']
        stage = args.stage
        if stage not in stage_choices:
            raise Exception("unkown stage.valid choices are {}".format(str(stage_choices)))
        try:
            with open(config_dir, "r") as stream:
                raw_dict = yaml.safe_load(stream)

                # general param
                self.random_seed = raw_dict['general']['random_seed']
                self.print_freq = raw_dict['general']['print_freq']
                self.resume = raw_dict['general']['resume']
                self.gpu = raw_dict['general']['gpu_id']
                self.arch = raw_dict['general']['arch']
                self.width_multiplier = raw_dict['general']['width_multiplier']
                self.workers = raw_dict['general']['workers']
                self.model_dir = raw_dict['general']['model_dir']
                self.log_dir = raw_dict['general']['log_dir']
                try:
                    self.init_func = raw_dict['general']['init_func']
                    self.smooth_eps = raw_dict['general']['smooth_eps']
                    self.alpha = raw_dict['general']['alpha']
                except:
                    print ('no fancy stuff for training')
                if stage == 'validate':
                    self.model = None
                    self.load_model = raw_dict[stage]['load_model']
                    try:
                        self.split_index = raw_dict[stage]['split_index']
                        self.noise_scale = raw_dict[stage]['noise_scale']
                        self.dcor_weighting = raw_dict[stage]['dcor_weighting']
                    except:
                        print('no dp')
                    return
                self.epochs = raw_dict[stage]['epochs']
                self.optimizer = raw_dict[stage]['optimizer']
                self.lr = float(raw_dict[stage]['lr'])
                self.lr_scheduler = raw_dict[stage]['lr_scheduler']
                self.save_model = raw_dict[stage]['save_model']
                if stage == 'pretrain':
                    self.aux = raw_dict[stage]['auxiliary']
                self.inv = (stage == 'inversion')
                self.adv = (stage == 'advtrain')
                self.dp = (stage == 'noisytrain')
                self.plot = False
                self.layers = None
                self.aux_layers = None
                self.load_model = None
                self.load_inv_model = None
                self.load_aux_model = None
                if stage != 'pretrain' and stage != 'onlyX' and stage != 'onlyM' and stage != 'Gan':
                    if 'layers' in raw_dict[stage]:
                        self.layers = raw_dict[stage]['layers']
                    if 'aux_layers' in raw_dict[stage]:
                        self.aux_layers = raw_dict[stage]['aux_layers']
                    if 'load_model' in raw_dict[stage]:
                        self.load_model = raw_dict[stage]['load_model']
                    if 'load_inv_model' in raw_dict[stage]:
                        self.load_inv_model = raw_dict[stage]['load_inv_model']
                    if 'load_aux_model' in raw_dict[stage]:
                        self.load_aux_model = raw_dict[stage]['load_aux_model']
                if stage == 'inversion' or stage == 'queryfree':
                    self.plot = raw_dict[stage]['plot']
                    if 'noise_scale' in raw_dict[stage] and 'split_index' in raw_dict[stage]:
                        self.split_index = raw_dict[stage]['split_index']
                        self.noise_scale = raw_dict[stage]['noise_scale']
                    if 'load_shadow_model' in raw_dict[stage]:
                        self.load_shadow_model = raw_dict[stage]['load_shadow_model']
                if stage == 'advtrain':
                    self.inv_lr = float(raw_dict[stage]['inv_lr'])
                    self.ckpt_interval = raw_dict[stage]['ckpt_interval']
                    self.num_steps = raw_dict[stage]['num_steps']
                    self.gamma = raw_dict[stage]['gamma']
                if stage == 'noisytrain':
                    self.split_index = raw_dict[stage]['split_index']
                    self.noise_scale = raw_dict[stage]['noise_scale']
                    self.dcor_weighting = raw_dict[stage]['dcor_weighting']
                if stage == 'antitrain':
                    self.inv_lr = float(raw_dict[stage]['inv_lr'])
                    self.plot = raw_dict[stage]['plot']
                    self.attn_loss = None
                    self.beta = float(raw_dict[stage]['beta'])
                    if 'noise_scale' in raw_dict[stage] and 'split_index' in raw_dict[stage]:
                        self.split_index = raw_dict[stage]['split_index']
                        self.noise_scale = raw_dict[stage]['noise_scale']
                if stage == 'onlyX' or stage == 'onlyM' or stage == 'Gan' or stage == 'queryfree':
                    if 'layers' in raw_dict[stage]:
                        self.layers = raw_dict[stage]['layers']
                    if 'load_model' in raw_dict[stage]:
                        self.load_model = raw_dict[stage]['load_model']
                    if 'load_shadow_model' in raw_dict[stage]:
                        self.load_shadow_model = raw_dict[stage]['load_shadow_model']
                    if 'lambda_TV' in raw_dict[stage]:
                        self.lambda_TV = raw_dict[stage]['lambda_TV']
                    if 'lambda_l2' in raw_dict[stage]:
                        self.lambda_l2 = raw_dict[stage]['lambda_l2'] 
                    if 'lambda_KLD' in raw_dict[stage]:
                        self.lambda_KLD = raw_dict[stage]['lambda_KLD']
                    if 'AMSGrad' in raw_dict[stage]:
                        self.AMSGrad = raw_dict[stage]['AMSGrad']
                    if 'inverse_num' in raw_dict[stage]:
                        self.inverse_num = raw_dict[stage]['inverse_num']
                    if 'noise_scale' in raw_dict[stage]:
                        self.noise_scale = raw_dict[stage]['noise_scale']
                    if 'split_index' in raw_dict[stage]:
                        self.split_index = raw_dict[stage]['split_index']
                if stage == 'Gan' or stage == 'queryfree':
                    if 'iter_z' in raw_dict[stage]:
                        self.iter_z= raw_dict[stage]['iter_z']
                    if 'iter_w' in raw_dict[stage]:
                        self.iter_w = raw_dict[stage]['iter_w']
                    if 'restarts' in raw_dict[stage]:
                        self.restarts = raw_dict[stage]['restarts']
                if stage == 'train_shadow':
                    self.load_model = raw_dict[stage]['load_model']
                    self.layers = raw_dict[stage]['layers']
                if stage == 'topaz':
                    self.lambda_mse = raw_dict[stage]['lambda_mse']
                try:
                    self.warmup_epochs = raw_dict[stage]['warmup_epochs']
                    self.warmup_lr = raw_dict[stage]['warmup_lr']
                except:
                    print ('no warmup for training')


                self.model = None
                self.inv_model = None
                self.aux_model = None
                self.trans_model = None
                self.shadow_model = None

        except yaml.YAMLError as exc:
            print(exc)    
