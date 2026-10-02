import numpy as np
import torch
import torch.nn.functional as F
import imageio

import os
from skimage.draw import disk

import matplotlib.pyplot as plt
import collections


class Logger:
    def __init__(self, log_dir, checkpoint_freq=100, visualizer_params=None, zfill_num=8, log_file_name='log.txt', vis_subdir=None):

        self.loss_list = []
        self.ckpt_dir = log_dir
        root_dir = os.getcwd()
        self.visualizations_dir = os.path.join(root_dir, vis_subdir if vis_subdir is not None else 'train-vis')
        if not os.path.exists(self.visualizations_dir):
            os.makedirs(self.visualizations_dir)
        self.log_file = open(os.path.join(root_dir, log_file_name), 'a')
        self.zfill_num = zfill_num
        self.visualizer = Visualizer(**visualizer_params)
        self.checkpoint_freq = checkpoint_freq
        self.epoch = 0
        self.names = None
        self.skip_full_checkpoint = False

    def log_scores(self, loss_names):
        loss_mean = np.array(self.loss_list).mean(axis=0)

        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(loss_names, loss_mean)])
        loss_string = str(self.epoch).zfill(self.zfill_num) + ") " + loss_string

        print(loss_string, file=self.log_file)
        self.loss_list = []
        self.log_file.flush()

    def visualize_rec(self, inp, out):
        image = self.visualizer.visualize(inp['driving'], inp['source'], out)
        imageio.imsave(os.path.join(self.visualizations_dir, "%s-rec.png" % str(self.epoch).zfill(self.zfill_num)), image)

    def save_checkpoint(self, emergent=False):
        # Skip if models not set (e.g., early exit) or path unwritable.
        if not hasattr(self, 'models') or self.models is None:
            return
        ckpt = {}
        for k, v in self.models.items():
            if hasattr(v, 'state_dict'):
                ckpt[k] = v.state_dict()
            elif torch.is_tensor(v):
                ckpt[k] = v.detach().cpu()
            else:
                continue
        ckpt['epoch'] = self.epoch
        ckpt_path = os.path.join(self.ckpt_dir, '%s-checkpoint.pth.tar' % str(self.epoch).zfill(self.zfill_num))
        if os.path.exists(ckpt_path) and emergent:
            return
        try:
            torch.save(ckpt, ckpt_path)
        except Exception as e:
            # Log to file and continue without crashing training
            try:
                print(f"Failed to save checkpoint {ckpt_path}: {e}", file=self.log_file)
                self.log_file.flush()
            except Exception:
                pass

    @staticmethod
    def load_checkpoint(checkpoint_path, generator=None, discriminator=None, kp_detector=None,
                        optimizer_generator=None, optimizer_discriminator=None, optimizer_kp_detector=None):
        if torch.cuda.is_available():
            map_location = None
        else:
            map_location = 'cpu'
        # Prefer safe loading when available (PyTorch >= 2.4)
        try:
            checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
        except TypeError:
            # Fallback for older torch versions
            checkpoint = torch.load(checkpoint_path, map_location)
        if generator is not None and 'generator' in checkpoint:
            generator.load_state_dict(checkpoint['generator'], strict=False)
        if kp_detector is not None and 'kp_detector' in checkpoint:
            kp_detector.load_state_dict(checkpoint['kp_detector'])

        # Return full checkpoint dict for accessing additional modules (e.g., style_encoder, style_to_kp_head)
        return checkpoint

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if getattr(self, 'skip_full_checkpoint', False):
            self.log_file.close()
            return
        if 'models' in self.__dict__:
            self.save_checkpoint()
        self.log_file.close()

    def log_iter(self, losses):
        losses = collections.OrderedDict(losses.items())
        if self.names is None:
            self.names = list(losses.keys())
        self.loss_list.append(list(losses.values()))

    def log_epoch(self, epoch, models, inp, out):
        self.epoch = epoch
        self.models = models
        if not getattr(self, 'skip_full_checkpoint', False) and (self.epoch + 1) % self.checkpoint_freq == 0:
            self.save_checkpoint()
        self.log_scores(self.names)
        if (
            not getattr(self, 'skip_train_visualizations', False)
            and ('prediction' in out)
            and ('driving' in inp)
        ):
            self.visualize_rec(inp, out)
        # Extra stats logging (lambda and KP errors if available)
        try:
            lam = out.get('lambda', None)
            if lam is not None:
                lam_val = float(lam.detach().cpu().item() if hasattr(lam, 'detach') else lam)
            else:
                lam_val = None

            kp_val_err = None
            kp_jac_err = None
            kp_val_l1 = None
            kp_jac_l1 = None
            if ('kp_hat' in out) and ('kp_driving_true' in out):
                v_hat = out['kp_hat']['value']
                v_true = out['kp_driving_true']['value']
                j_hat = out['kp_hat']['jacobian']
                j_true = out['kp_driving_true']['jacobian']
                kp_val_err = float(((v_hat - v_true) ** 2).mean().detach().cpu().item())
                kp_jac_err = float(((j_hat - j_true) ** 2).mean().detach().cpu().item())
                kp_val_l1 = float((torch.abs(v_hat - v_true).mean()).detach().cpu().item())
                kp_jac_l1 = float((torch.abs(j_hat - j_true).mean()).detach().cpu().item())

            msg_parts = [str(self.epoch).zfill(self.zfill_num) + ')']
            if lam_val is not None:
                msg_parts.append(f"lambda={lam_val*100:.1f}%")
            if kp_val_err is not None:
                msg_parts.append(f"kp_value_l2={kp_val_err:.6f}")
            if kp_jac_err is not None:
                msg_parts.append(f"kp_jac_fro={kp_jac_err:.6f}")
            if kp_val_l1 is not None:
                msg_parts.append(f"kp_value_l1={kp_val_l1:.6f}")
            if kp_jac_l1 is not None:
                msg_parts.append(f"kp_jac_l1={kp_jac_l1:.6f}")
            if len(msg_parts) > 1:
                print(' '.join(msg_parts), file=self.log_file)
                self.log_file.flush()

            # Log Transformer KP regression losses for monitoring student-teacher alignment
            if ('kp_hat' in out) and ('kp_driving_true' in out):
                v_hat = out['kp_hat']['value']
                v_true = out['kp_driving_true']['value']
                jac_hat = out['kp_hat']['jacobian']
                jac_true = out['kp_driving_true']['jacobian']
                kp_val_l2 = float(((v_hat - v_true) ** 2).mean().detach().cpu().item())
                kp_jac_l2 = float(((jac_hat - jac_true) ** 2).mean().detach().cpu().item())
                kp_val_l1 = float((torch.abs(v_hat - v_true).mean()).detach().cpu().item())
                kp_jac_l1 = float((torch.abs(jac_hat - jac_true).mean()).detach().cpu().item())
                print(f"  kp_val_l2={kp_val_l2:.6f} kp_jac_l2={kp_jac_l2:.6f} kp_val_l1={kp_val_l1:.6f} kp_jac_l1={kp_jac_l1:.6f}", file=self.log_file)
                self.log_file.flush()
        except Exception:
            pass


class Visualizer:
    def __init__(self, kp_size=5, draw_border=False, colormap='gist_rainbow'):
        self.kp_size = kp_size
        self.draw_border = draw_border
        self.colormap = plt.get_cmap(colormap)

    @staticmethod
    def _to_numpy(tensor):
        """
        Convert torch tensor to float32 numpy for safe visualization.
        """
        if torch.is_tensor(tensor):
            return tensor.detach().to(torch.float32).cpu().numpy()
        return np.asarray(tensor)

    @staticmethod
    def flow_to_color(flow, max_flow=None):
        """
        Convert optical flow field (H, W, 2) to RGB color map (H, W, 3).
        Uses HSV color space: Hue encodes direction, Saturation & Value encode magnitude.
        """
        h, w = flow.shape[:2]
        fx, fy = flow[:, :, 0], flow[:, :, 1]
        
        # Calculate angle (direction) and magnitude (size)
        ang = np.arctan2(fy, fx) + np.pi
        mag = np.sqrt(fx**2 + fy**2)
        
        # Normalize magnitude
        if max_flow is None:
            max_flow = mag.max() + 1e-6
        mag = np.clip(mag / max_flow, 0, 1)
        
        # HSV: H=direction, S=1, V=magnitude
        hsv = np.zeros((h, w, 3), dtype=np.float32)
        hsv[:, :, 0] = ang / (2 * np.pi)  # Hue: 0-1
        hsv[:, :, 1] = 1.0                 # Saturation: fixed at 1
        hsv[:, :, 2] = mag                 # Value: magnitude
        
        # HSV to RGB
        from matplotlib.colors import hsv_to_rgb
        rgb = hsv_to_rgb(hsv)
        return rgb

    @staticmethod
    def flow_to_grid(flow, grid_spacing=16):
        """
        Visualize optical flow as deformed grid lines.
        Shows how a regular grid is warped by the flow field.
        
        Args:
            flow: (H, W, 2) optical flow field
            grid_spacing: spacing between grid lines in pixels
        
        Returns:
            (H, W, 3) RGB image with colored grid lines on black background
        """
        h, w = flow.shape[:2]
        
        # Create black background
        grid_img = np.zeros((h, w, 3), dtype=np.float32)
        
        # Original grid coordinates
        y_coords = np.arange(0, h, grid_spacing)
        x_coords = np.arange(0, w, grid_spacing)
        
        # Draw horizontal lines (warped by flow)
        for y in y_coords:
            if y >= h:
                continue
            x_line = np.arange(w)
            y_line = np.ones(w, dtype=int) * y
            
            # Apply displacement
            x_warped = np.clip(x_line + flow[y_line, x_line, 0] * w / 2, 0, w - 1).astype(int)
            y_warped = np.clip(y_line + flow[y_line, x_line, 1] * h / 2, 0, h - 1).astype(int)
            
            # Draw line segments with color indicating displacement
            for i in range(len(x_warped) - 1):
                try:
                    # Calculate displacement magnitude at this point
                    displacement = np.sqrt(flow[y, x_line[i], 0]**2 + flow[y, x_line[i], 1]**2)
                    color = plt.cm.jet(np.clip(displacement * 5, 0, 1))[:3]  # Red = large displacement
                    
                    # Draw line segment
                    rr, cc = np.array([y_warped[i], y_warped[i+1]]), np.array([x_warped[i], x_warped[i+1]])
                    valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
                    grid_img[rr[valid], cc[valid]] = color
                except:
                    pass
        
        # Draw vertical lines
        for x in x_coords:
            if x >= w:
                continue
            y_line = np.arange(h)
            x_line = np.ones(h, dtype=int) * x
            
            # Apply displacement
            x_warped = np.clip(x_line + flow[y_line, x_line, 0] * w / 2, 0, w - 1).astype(int)
            y_warped = np.clip(y_line + flow[y_line, x_line, 1] * h / 2, 0, h - 1).astype(int)
            
            for i in range(len(y_warped) - 1):
                try:
                    displacement = np.sqrt(flow[y_line[i], x, 0]**2 + flow[y_line[i], x, 1]**2)
                    color = plt.cm.jet(np.clip(displacement * 5, 0, 1))[:3]
                    
                    rr, cc = np.array([y_warped[i], y_warped[i+1]]), np.array([x_warped[i], x_warped[i+1]])
                    valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
                    grid_img[rr[valid], cc[valid]] = color
                except:
                    pass
        
        return grid_img

    @staticmethod
    def flow_to_magnitude_heatmap(flow):
        """
        Convert flow field to displacement magnitude heatmap.
        
        Args:
            flow: (H, W, 2) optical flow field
        
        Returns:
            (H, W, 3) RGB heatmap, blue=small displacement, red=large displacement
        """
        # Calculate displacement magnitude
        magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
        
        # Normalize to 0-1
        mag_norm = magnitude / (magnitude.max() + 1e-6)
        
        # Use jet colormap: blue (small) -> green -> yellow -> red (large)
        heatmap = plt.cm.jet(mag_norm)[:, :, :3]
        
        return heatmap

    @staticmethod
    def flow_with_arrows(background_img, flow, arrow_spacing=16, arrow_scale=1.0):
        """
        Overlay arrows on background image to show displacement direction.
        
        Args:
            background_img: (H, W, 3) background image
            flow: (H, W, 2) optical flow field
            arrow_spacing: spacing between arrows
            arrow_scale: arrow length scaling factor
        
        Returns:
            (H, W, 3) image with arrows overlaid
        """
        import cv2
        
        img = (background_img * 255).astype(np.uint8).copy()
        h, w = flow.shape[:2]
        
        # Draw arrows on grid points
        for y in range(0, h, arrow_spacing):
            for x in range(0, w, arrow_spacing):
                # Displacement vector (convert to pixel coordinates)
                dx = flow[y, x, 0] * w / 2 * arrow_scale
                dy = flow[y, x, 1] * h / 2 * arrow_scale
                
                # Calculate displacement magnitude
                magnitude = np.sqrt(dx**2 + dy**2)
                
                # Only draw arrows with significant displacement
                if magnitude > 1.0:
                    # Color based on magnitude: green (small) -> yellow -> red (large)
                    color_val = np.clip(magnitude / 20.0, 0, 1)
                    color = (0, int(255 * (1 - color_val)), int(255 * color_val))  # BGR
                    
                    # Start and end points
                    pt1 = (int(x), int(y))
                    pt2 = (int(x + dx), int(y + dy))
                    
                    # Draw arrow
                    try:
                        cv2.arrowedLine(img, pt1, pt2, color, 1, tipLength=0.3)
                    except:
                        pass
        
        return img.astype(np.float32) / 255.0

    def draw_image_with_kp(self, image, kp_array):
        image = np.copy(image)
        spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]
        kp_array = spatial_size * (kp_array + 1) / 2
        num_kp = kp_array.shape[0]
        for kp_ind, kp in enumerate(kp_array):
            try:
                # New skimage version (>= 0.19)
                rr, cc = disk((kp[1], kp[0]), self.kp_size, shape=image.shape[:2])
            except TypeError:
                # Old skimage version
                rr, cc = disk(kp[1], kp[0], self.kp_size, shape=image.shape[:2])
            image[rr, cc] = np.array(self.colormap(kp_ind / num_kp))[:3]
        return image

    def create_image_column_with_kp(self, images, kp):
        image_array = np.array([self.draw_image_with_kp(v, k) for v, k in zip(images, kp)])
        return self.create_image_column(image_array)

    def create_image_column(self, images):
        if self.draw_border:
            images = np.copy(images)
            images[:, :, [0, -1]] = (1, 1, 1)
        return np.concatenate(list(images), axis=0)

    def create_image_grid(self, *args):
        out = []
        for arg in args:
            if type(arg) == tuple:
                out.append(self.create_image_column_with_kp(arg[0], arg[1]))
            else:
                out.append(self.create_image_column(arg))
        grid = np.concatenate(out, axis=1)
        return grid

    def visualize(self, driving, source, out):
        images = []

        # Source image with keypoints
        source = self._to_numpy(source)
        kp_source = self._to_numpy(out['kp_source']['value'])
        source = np.transpose(source, [0, 2, 3, 1])
        images.append((source, kp_source))

        # Equivariance visualization
        if 'transformed_frame' in out:
            transformed = self._to_numpy(out['transformed_frame'])
            transformed = np.transpose(transformed, [0, 2, 3, 1])
            transformed_kp = self._to_numpy(out['transformed_kp']['value'])
            images.append((transformed, transformed_kp))

        # Driving image with keypoints (student prediction vs teacher ground truth, shown side-by-side for comparison)
        kp_driving = self._to_numpy(out['kp_driving']['value'])
        driving = self._to_numpy(driving)
        driving = np.transpose(driving, [0, 2, 3, 1])
        images.append((driving, kp_driving))
        if 'kp_driving_true' in out:
            kp_driving_true = self._to_numpy(out['kp_driving_true']['value'])
            images.append((driving, kp_driving_true))

        # Deformed image
        if 'deformed' in out:
            deformed = self._to_numpy(out['deformed'])
            deformed = np.transpose(deformed, [0, 2, 3, 1])
            images.append(deformed)

        # Result with and without keypoints
        prediction = self._to_numpy(out['prediction'])
        prediction = np.transpose(prediction, [0, 2, 3, 1])
        if 'kp_norm' in out:
            kp_norm = self._to_numpy(out['kp_norm']['value'])
            images.append((prediction, kp_norm))
        images.append(prediction)

        # Teacher prediction (if provided)
        if 'teacher_prediction' in out:
            tpred = self._to_numpy(out['teacher_prediction'])
            tpred = np.transpose(tpred, [0, 2, 3, 1])
            images.append(tpred)


        ## Occlusion map
        if 'occlusion_map' in out:
            occlusion_map = out['occlusion_map'].detach().to(torch.float32).cpu().repeat(1, 3, 1, 1)
            occlusion_map = F.interpolate(occlusion_map, size=source.shape[1:3]).detach().cpu().numpy()
            occlusion_map = np.transpose(occlusion_map, [0, 2, 3, 1])
            images.append(occlusion_map)

        ## Optical flow field visualization (using multiple methods)
        if 'deformation' in out:
            # Student flow field
            student_flow = self._to_numpy(out['deformation'])  # (B, H, W, 2)
            
            # Get target size (matching source image)
            target_h, target_w = source.shape[1:3]
            
            from skimage.transform import resize
            
            # Method 1: Heatmap (displacement magnitude) - Most intuitive
            student_heatmap_list = []
            for i in range(student_flow.shape[0]):
                heatmap = self.flow_to_magnitude_heatmap(student_flow[i])
                if heatmap.shape[0] != target_h or heatmap.shape[1] != target_w:
                    heatmap = resize(heatmap, (target_h, target_w), anti_aliasing=True)
                student_heatmap_list.append(heatmap)
            student_heatmap_vis = np.array(student_heatmap_list)
            images.append(student_heatmap_vis)
            
            # Method 2: Deformed grid - Most clear
            student_grid_list = []
            for i in range(student_flow.shape[0]):
                grid = self.flow_to_grid(student_flow[i], grid_spacing=16)
                if grid.shape[0] != target_h or grid.shape[1] != target_w:
                    grid = resize(grid, (target_h, target_w), anti_aliasing=True)
                student_grid_list.append(grid)
            student_grid_vis = np.array(student_grid_list)
            images.append(student_grid_vis)
            
            # If teacher flow field is available, visualize it too
            if 'teacher_deformation' in out:
                teacher_flow = self._to_numpy(out['teacher_deformation'])
                
                # Teacher heatmap
                teacher_heatmap_list = []
                for i in range(teacher_flow.shape[0]):
                    heatmap = self.flow_to_magnitude_heatmap(teacher_flow[i])
                    if heatmap.shape[0] != target_h or heatmap.shape[1] != target_w:
                        heatmap = resize(heatmap, (target_h, target_w), anti_aliasing=True)
                    teacher_heatmap_list.append(heatmap)
                teacher_heatmap_vis = np.array(teacher_heatmap_list)
                images.append(teacher_heatmap_vis)
                
                # Teacher grid
                teacher_grid_list = []
                for i in range(teacher_flow.shape[0]):
                    grid = self.flow_to_grid(teacher_flow[i], grid_spacing=16)
                    if grid.shape[0] != target_h or grid.shape[1] != target_w:
                        grid = resize(grid, (target_h, target_w), anti_aliasing=True)
                    teacher_grid_list.append(grid)
                teacher_grid_vis = np.array(teacher_grid_list)
                images.append(teacher_grid_vis)
                
                # Difference heatmap
                flow_diff = student_flow - teacher_flow
                diff_heatmap_list = []
                for i in range(flow_diff.shape[0]):
                    heatmap = self.flow_to_magnitude_heatmap(flow_diff[i])
                    if heatmap.shape[0] != target_h or heatmap.shape[1] != target_w:
                        heatmap = resize(heatmap, (target_h, target_w), anti_aliasing=True)
                    diff_heatmap_list.append(heatmap)
                diff_heatmap_vis = np.array(diff_heatmap_list)
                images.append(diff_heatmap_vis)

        # Simplified: only keep key columns, no longer expand each sparse deformation visualization

        image = self.create_image_grid(*images)

        # Add simple text header row labeling columns (source, transformed?, driving, deformed, prediction, teacher)
        try:
            import cv2
            header_h = 24
            h, w, c = image.shape
            canvas = np.ones((h + header_h, w, c), dtype=image.dtype)
            canvas[header_h:] = image
            # Column labels estimation by counting added blocks
            labels = []
            # 1: source+kp
            labels.append('source+kp')
            idx = 1
            # maybe transformed+kp
            if 'transformed_frame' in out:
                labels.append('equiv+kp')
                idx += 1
            # driving+kp
            labels.append('driving+kp')
            if 'kp_driving_true' in out:
                labels.append('teacher_kp')
            # deformed
            if 'deformed' in out:
                labels.append('deformed')
            # prediction
            labels.append('student_pred')
            # teacher
            if 'teacher_prediction' in out:
                labels.append('teacher_pred')
            # occlusion
            if 'occlusion_map' in out:
                labels.append('occlusion')
            # flow (heatmap + grid)
            if 'deformation' in out:
                labels.append('S_heat')  # Student heatmap
                labels.append('S_grid')  # Student grid
                if 'teacher_deformation' in out:
                    labels.append('T_heat')  # Teacher heatmap
                    labels.append('T_grid')  # Teacher grid
                    labels.append('diff')    # Difference

            # compute each block width by splitting equally
            step = w // len(labels)
            for i, text in enumerate(labels):
                cv2.putText(canvas, text, (i * step + 5, header_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            image = canvas
        except Exception:
            pass

        image = (255 * image).astype(np.uint8)
        return image

