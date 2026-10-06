from torch import nn
import torch
import torch.nn.functional as F
from modules.util import AntiAliasInterpolation2d, make_coordinate_grid
from torchvision import models
import numpy as np
from torch.autograd import grad


class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss. See Sec 3.3.
    """
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class ImagePyramid(torch.nn.Module):
    """
    Create image pyramid for computing pyramid perceptual loss. See Sec 3.3
    """
    def __init__(self, scales, num_channels):
        super(ImagePyramid, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict


class Transform:
    """
    Random tps transformation for equivariance constraints. See Sec 3.3
    """
    def __init__(self, bs, **kwargs):
        noise = torch.normal(mean=0, std=kwargs['sigma_affine'] * torch.ones([bs, 2, 3]))
        self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
        self.bs = bs

        if ('sigma_tps' in kwargs) and ('points_tps' in kwargs):
            self.tps = True
            self.control_points = make_coordinate_grid((kwargs['points_tps'], kwargs['points_tps']), type=noise.type())
            self.control_points = self.control_points.unsqueeze(0)
            self.control_params = torch.normal(mean=0,
                                               std=kwargs['sigma_tps'] * torch.ones([bs, 1, kwargs['points_tps'] ** 2]))
        else:
            self.tps = False

    def transform_frame(self, frame):
        grid = make_coordinate_grid(frame.shape[2:], type=frame.type()).unsqueeze(0)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        grid = self.warp_coordinates(grid).view(self.bs, frame.shape[2], frame.shape[3], 2)
        return F.grid_sample(frame, grid, padding_mode="reflection", align_corners=False)

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type())
        theta = theta.unsqueeze(1)
        transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
        transformed = transformed.squeeze(-1)

        if self.tps:
            control_points = self.control_points.type(coordinates.type())
            control_params = self.control_params.type(coordinates.type())
            distances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
            distances = torch.abs(distances).sum(-1)

            result = distances ** 2
            result = result * torch.log(distances + 1e-6)
            result = result * control_params
            result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result

        return transformed

    def jacobian(self, coordinates):
        new_coordinates = self.warp_coordinates(coordinates)
        grad_x = grad(new_coordinates[..., 0].sum(), coordinates, create_graph=True)
        grad_y = grad(new_coordinates[..., 1].sum(), coordinates, create_graph=True)
        jacobian = torch.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], dim=-2)
        return jacobian


def detach_kp(kp):
    return {key: value.detach() for key, value in kp.items()}


class GeneratorFullModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params):
        super(GeneratorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = train_params['scales']
        self.disc_scales = self.discriminator.scales
        self.pyramid = ImagePyramid(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()
            if torch.cuda.is_available():
                self.vgg = self.vgg.cuda()

    def forward(self, x):
        kp_source = self.kp_extractor(x['source'])
        kp_driving = self.kp_extractor(x['driving'])

        generated = self.generator(x['source'], kp_source=kp_source, kp_driving=kp_driving)
        generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})

        loss_values = {}

        pyramid_real = self.pyramid(x['driving'])
        pyramid_generated = self.pyramid(generated['prediction'])

        if sum(self.loss_weights['perceptual']) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramid_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramid_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual'][i] * value
                loss_values['perceptual'] = value_total

        if self.loss_weights['generator_gan'] != 0:
            discriminator_maps_generated = self.discriminator(pyramid_generated, kp=detach_kp(kp_driving))
            discriminator_maps_real = self.discriminator(pyramid_real, kp=detach_kp(kp_driving))
            value_total = 0
            for scale in self.disc_scales:
                key = 'prediction_map_%s' % scale
                value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                value_total += self.loss_weights['generator_gan'] * value
            loss_values['gen_gan'] = value_total

            if sum(self.loss_weights['feature_matching']) != 0:
                value_total = 0
                for scale in self.disc_scales:
                    key = 'feature_maps_%s' % scale
                    for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                        if self.loss_weights['feature_matching'][i] == 0:
                            continue
                        value = torch.abs(a - b).mean()
                        value_total += self.loss_weights['feature_matching'][i] * value
                    loss_values['feature_matching'] = value_total

        if (self.loss_weights['equivariance_value'] + self.loss_weights['equivariance_jacobian']) != 0:
            transform = Transform(x['driving'].shape[0], **self.train_params['transform_params'])
            transformed_frame = transform.transform_frame(x['driving'])
            transformed_kp = self.kp_extractor(transformed_frame)

            generated['transformed_frame'] = transformed_frame
            generated['transformed_kp'] = transformed_kp

            ## Value loss part
            if self.loss_weights['equivariance_value'] != 0:
                value = torch.abs(kp_driving['value'] - transform.warp_coordinates(transformed_kp['value'])).mean()
                loss_values['equivariance_value'] = self.loss_weights['equivariance_value'] * value

            ## jacobian loss part
            if self.loss_weights['equivariance_jacobian'] != 0:
                jacobian_transformed = torch.matmul(transform.jacobian(transformed_kp['value']),
                                                    transformed_kp['jacobian'])

                normed_driving = torch.inverse(kp_driving['jacobian'])
                normed_transformed = jacobian_transformed
                value = torch.matmul(normed_driving, normed_transformed)

                eye = torch.eye(2).view(1, 1, 2, 2).type(value.type())

                value = torch.abs(eye - value).mean()
                loss_values['equivariance_jacobian'] = self.loss_weights['equivariance_jacobian'] * value

        return loss_values, generated


class DiscriminatorFullModel(torch.nn.Module):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params):
        super(DiscriminatorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = self.discriminator.scales
        self.pyramid = ImagePyramid(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

    def forward(self, x, generated):
        pyramid_real = self.pyramid(x['driving'])
        pyramid_generated = self.pyramid(generated['prediction'].detach())

        kp_driving = generated['kp_driving']
        font_labels = x.get('font_id', None)
        if font_labels is not None and torch.is_tensor(font_labels):
            # convert to 0-based long
            font_labels = font_labels.to(pyramid_real[next(iter(pyramid_real))].device)
            if font_labels.dtype != torch.long:
                font_labels = font_labels.long()
            font_labels = torch.clamp_min(font_labels - 1, 0)
        discriminator_maps_generated = self.discriminator(pyramid_generated, kp=detach_kp(kp_driving), y=font_labels)
        discriminator_maps_real = self.discriminator(pyramid_real, kp=detach_kp(kp_driving), y=font_labels)

        loss_values = {}
        value_total = 0
        for scale in self.scales:
            key = 'prediction_map_%s' % scale
            value = (1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2
            value_total += self.loss_weights['discriminator_gan'] * value.mean()
        loss_values['disc_gan'] = value_total

        return loss_values


class StudentTeacherFontModel(torch.nn.Module):
    """
    Student-teacher distillation model for font stylization.
    """

    def __init__(self, kp_extractor, teacher_generator, font_student, train_params):
        super(StudentTeacherFontModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.teacher_generator = teacher_generator
        self.font_student = font_student
        self.train_params = train_params
        self.scales = train_params.get('scales', [1])
        self.loss_weights = train_params.get('loss_weights', {})
        self.epoch = 0
        self.pyramid = None
        num_channels = getattr(teacher_generator, 'num_channels', 3) if teacher_generator is not None else 3
        self.pyramid = ImagePyramid(self.scales, num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.use_perc = 'perceptual' in self.loss_weights and sum(self.loss_weights['perceptual']) != 0
        if self.use_perc:
            self.vgg = Vgg19()
            if torch.cuda.is_available():
                self.vgg = self.vgg.cuda()

    @staticmethod
    def tv_loss(flow):
        dx = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :]).mean()
        dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
        return dx + dy

    @staticmethod
    def _match_flow_size(flow_bhw2, target_h, target_w):
        if flow_bhw2.shape[1] == target_h and flow_bhw2.shape[2] == target_w:
            return flow_bhw2
        f = flow_bhw2.permute(0, 3, 1, 2)
        f = torch.nn.functional.interpolate(f, size=(target_h, target_w), mode='bilinear', align_corners=False)
        return f.permute(0, 2, 3, 1)

    @staticmethod
    def _match_map_size(map_b1hw, target_h, target_w):
        if map_b1hw.shape[2] == target_h and map_b1hw.shape[3] == target_w:
            return map_b1hw
        return torch.nn.functional.interpolate(map_b1hw, size=(target_h, target_w), mode='bilinear', align_corners=False)

    def _get_loss_weight_multiplier(self, loss_name):
        """
        Dynamically adjust loss weights based on training stage (optional).
        """
        sched = self.train_params.get('loss_weight_schedule', None)
        if sched is None:
            return 1.0
        num_epochs = self.train_params.get('num_epochs', 100)
        warmup_ratio = sched.get('warmup_ratio', 0.1)
        warmup_epochs = max(1, int(num_epochs * warmup_ratio))
        kp_loss_names = sched.get('kp_loss_names', ['kp_value_reg', 'kp_jac_reg'])
        if self.epoch < warmup_epochs:
            if loss_name in kp_loss_names:
                return 1.0
            else:
                return 0.0
        else:
            return 1.0

    def forward(self, x):

        teacher_gen = {}
        if self.teacher_generator is not None:
            with torch.no_grad():
                kp_source_t = self.kp_extractor(x['source'])
                kp_driving_t = self.kp_extractor(x['driving'])
                teacher_gen = self.teacher_generator(x['source'], kp_source=kp_source_t, kp_driving=kp_driving_t)

        if hasattr(self.font_student, 'epoch'):
            self.font_student.epoch = getattr(self, 'epoch', 0)
        student_gen = self.font_student(x)
        if 'prediction' in teacher_gen:
            student_gen['teacher_prediction'] = teacher_gen['prediction']
        if 'deformation' in teacher_gen:
            student_gen['teacher_deformation'] = teacher_gen['deformation']

        loss_values = {}

        if 'l1' in self.loss_weights and self.loss_weights['l1'] != 0:
            l1 = torch.abs(student_gen['prediction'] - x['driving']).mean()
            loss_values['l1'] = self.loss_weights['l1'] * l1 * self._get_loss_weight_multiplier('l1')
        if 'mid_keypoint' in student_gen:
            loss_values['mid_keypoint'] = student_gen['mid_keypoint'] * self._get_loss_weight_multiplier('mid_keypoint')

        if self.use_perc:
            pyramid_real = self.pyramid(x['driving'])
            pyramid_generated = self.pyramid(student_gen['prediction'])
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramid_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramid_real['prediction_' + str(scale)])
                for i, weight in enumerate(self.loss_weights['perceptual']):
                    if weight == 0:
                        continue
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += weight * value
            loss_values['perceptual'] = value_total * self._get_loss_weight_multiplier('perceptual')

        if 'tv' in self.loss_weights and self.loss_weights['tv'] != 0 and ('deformation' in student_gen):
            tv = self.tv_loss(student_gen['deformation'])
            loss_values['tv'] = self.loss_weights['tv'] * tv * self._get_loss_weight_multiplier('tv')

        if self.loss_weights.get('pred_distill', 0) != 0 and ('prediction' in teacher_gen):
            pd = torch.abs(student_gen['prediction'] - teacher_gen['prediction']).mean()
            loss_values['pred_distill'] = self.loss_weights['pred_distill'] * pd * self._get_loss_weight_multiplier('pred_distill')

        if self.loss_weights.get('flow_distill', 0) != 0 and ('deformation' in teacher_gen) and ('deformation' in student_gen):
            s_flow = student_gen['deformation']
            t_flow = teacher_gen['deformation']
            t_h, t_w = t_flow.shape[1], t_flow.shape[2]
            s_flow = self._match_flow_size(s_flow, t_h, t_w)
            fd = torch.mean((s_flow - t_flow) ** 2)
            flow_w = self.loss_weights['flow_distill']
            loss_values['flow_distill'] = flow_w * fd * self._get_loss_weight_multiplier('flow_distill')

        if self.loss_weights.get('occ_distill', 0) != 0 and ('occlusion_map' in teacher_gen) and ('occlusion_map' in student_gen):
            s_occ = student_gen['occlusion_map']
            t_occ = teacher_gen['occlusion_map']
            t_h, t_w = t_occ.shape[2], t_occ.shape[3]
            s_occ = self._match_map_size(s_occ, t_h, t_w)
            od = torch.nn.functional.binary_cross_entropy(s_occ, t_occ)
            loss_values['occ_distill'] = self.loss_weights['occ_distill'] * od * self._get_loss_weight_multiplier('occ_distill')

        if 'kp_hat' in student_gen and 'kp_driving_true' in student_gen:
            # Optional experiment-level multiplier for the complete geometric
            # block. It defaults to 1.0, so existing configurations and
            # checkpoints retain exactly the original objective.
            geo_group = float(self.loss_weights.get('geo_group', 1.0))
            if self.loss_weights.get('kp_value_reg', 0) != 0:
                kv = torch.mean((student_gen['kp_hat']['value'] - student_gen['kp_driving_true']['value']) ** 2)
                loss_values['kp_value_reg'] = geo_group * self.loss_weights['kp_value_reg'] * kv * self._get_loss_weight_multiplier('kp_value_reg')
            l1_val = torch.abs(student_gen['kp_hat']['value'] - student_gen['kp_driving_true']['value']).mean()
            loss_values['kp_l1'] = geo_group * l1_val
            if self.loss_weights.get('kp_jac_reg', 0) != 0:
                kj = torch.mean((student_gen['kp_hat']['jacobian'] - student_gen['kp_driving_true']['jacobian']) ** 2)
                loss_values['kp_jac_reg'] = geo_group * self.loss_weights['kp_jac_reg'] * kj * self._get_loss_weight_multiplier('kp_jac_reg')
            if ('jacobian' in student_gen['kp_hat']) and ('jacobian' in student_gen['kp_driving_true']):
                jac_l1 = torch.abs(student_gen['kp_hat']['jacobian'] - student_gen['kp_driving_true']['jacobian']).mean()
                loss_values['kp_jac_l1'] = geo_group * jac_l1

        kp_equiv_w = self.loss_weights.get('kp_equiv', 0)
        kp_equiv_cfg = self.train_params.get('kp_equiv', None)
        if (kp_equiv_w != 0) and (kp_equiv_cfg is not None):
            try:
                sigma_affine = float(kp_equiv_cfg.get('sigma_affine', 0.02))
                bs = x['driving'].shape[0]
                theta_noise = torch.normal(mean=0, std=sigma_affine * torch.ones([bs, 2, 3], device=x['driving'].device))
                theta = theta_noise + torch.eye(2, 3, device=x['driving'].device).view(1, 2, 3)
                grid = torch.nn.functional.affine_grid(theta, x['driving'].size(), align_corners=False)
                driving_t = torch.nn.functional.grid_sample(x['driving'], grid, padding_mode='reflection', align_corners=False)

                x_t = dict(x)
                x_t['driving'] = driving_t
                student_gen_t = self.font_student(x_t)
                if ('kp_hat' in student_gen) and ('kp_hat' in student_gen_t):
                    v = student_gen['kp_hat']['value']
                    j = student_gen['kp_hat']['jacobian']
                    A = theta[:, :, :2]
                    b = theta[:, :, 2:]
                    v_flat = v.view(v.shape[0], v.shape[1], 2, 1)
                    v_warp = torch.matmul(A.unsqueeze(1), v_flat).squeeze(-1) + b.unsqueeze(1)
                    j_warp = torch.matmul(A.unsqueeze(1).unsqueeze(1), j).squeeze(1)
                    kv_e = torch.mean((student_gen_t['kp_hat']['value'] - v_warp) ** 2)
                    kj_e = torch.mean((student_gen_t['kp_hat']['jacobian'] - j_warp) ** 2)
                    loss_values['kp_equiv'] = kp_equiv_w * (kv_e + kj_e) * self._get_loss_weight_multiplier('kp_equiv')
            except Exception:
                pass

        return loss_values, student_gen
