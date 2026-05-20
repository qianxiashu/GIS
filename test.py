import argparse, os
import secrets

import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
import copy

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

import torchvision.transforms as transforms
import torch.nn.functional as F
import time
import pickle
import random

feat_maps = []

class SecretToTensor:
    def __init__(self, secret, lambda_, nz, mean=0, std=1, tau=0.5):
        self.secret = secret
        self.lambda_ = lambda_
        self.nz = nz
        self.mean = mean
        self.std = std
        self.tau = tau

    def secret_transform(self):
        m = []
        len_s = len(self.secret)
        if len_s % self.lambda_ != 0:
            num_of_supplement_bits = self.lambda_ - len_s % self.lambda_
            self.secret += '0' * num_of_supplement_bits
        len_of_m = len(self.secret) // self.lambda_
        for i in range(len_of_m):
            m.append(self.secret[self.lambda_ * i:self.lambda_ * i + self.lambda_])
        return m


    def secret_mapping(self):
        m = self.secret_transform()
        noise = np.empty([1, self.nz])
        special_positions = []  # 用于记录满足条件的位置
        b_index = 0  # 用于跟踪 b_i 的位置

        # 生成完整的噪声
        for i in range(self.nz):
            u1, u2 = np.random.rand(2)
            random_noise = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
            noise[0][i] = random_noise

        # 检查并调整噪声
        for i in range(self.nz):
            if i >= 239 and i % 240 == 0:
            # if i % 256 == 0:
                if abs(noise[0][i]) < 0:
                    # 寻找合适的替换位置
                    found = False
                    min_diff = float('inf')
                    best_j = -1
                    for j in range(self.nz):
                        if j != i and j % 240 != 0 and abs(noise[0][j]) > 0:  # 修改条件，允许所有位置
                            diff = abs(noise[0][i] - noise[0][j])
                            if diff < min_diff:
                                min_diff = diff
                                best_j = j

                    # 如果找到合适的替换，进行交换
                    if best_j != -1:
                        noise[0][i], noise[0][best_j] = noise[0][best_j], noise[0][i]
                        found = True

                    # 如果没有找到合适的替换，重新生成
                    while not found:
                        u1, u2 = np.random.rand(2)
                        new_noise = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
                        if abs(new_noise) > 0:
                            noise[0][i] = new_noise
                            found = True

                # 根据 b_i 的值调整噪声
                if b_index < len(m):  # 确保索引不超出范围
                    b_i = int(m[b_index])
                    if b_i == 1:
                        noise[0][i] = self.mean + self.std * np.abs(noise[0][i])
                    else:
                        noise[0][i] = self.mean - self.std * np.abs(noise[0][i])
                    b_index += 1

        # print(f"Special positions: {special_positions}")
        # print(f"Final b_index: {b_index}")  # 输出最终的 b_index 值

        # 仅返回前 b_index 位的秘密信息
        truncated_secret = self.secret[:b_index]

        return noise, special_positions, truncated_secret,b_index


    def bin_to_inter(self, bin_str):
        return int(bin_str, 2)

    def generate_tensor(self, target_shape):
        noise, _, final_secret = self.secret_mapping()  # 获取最终的秘密信息
        noise_tensor = torch.tensor(noise, dtype=torch.float32)
        noise_tensor = noise_tensor.view(target_shape)
        return noise_tensor, final_secret  # 返回张量和最终的秘密信息


class TensorToSecret:
    def __init__(self, noise_tensor, lambda_, bits_to_extract=None, mean=0, std=1, tau=0.5):
        self.noise_tensor = noise_tensor
        self.lambda_ = lambda_
        self.mean = mean
        self.std = std
        self.tau = tau
        self.bits_to_extract = bits_to_extract  # 新增参数

    def tensor_to_secret(self, special_positions):
        # 将噪声张量展平并转换为 numpy 数组
        noise = self.noise_tensor.cpu().view(-1).numpy()

        # 初始化一个空列表来存储二进制值
        m = []


        # 每隔 5 个位置提取一次
        for i in range(240 ,len(noise) , 240):
        # for i in range(0, len(noise), 256):
            value = noise[i]
            # 根据值恢复二进制位
            if value > self.mean:  # 大于均值
                m.append('1')
            else:  # 小于等于均值
                m.append('0')

        # 将列表转换为字符串
        secret_binary = ''.join(m)

        # 清理多余的补位
        # 这里假设补位为 '0'
        num_of_bits = len(secret_binary)
        if num_of_bits % self.lambda_ != 0:
            # 计算补位数量
            num_of_supplement_bits = self.lambda_ - num_of_bits % self.lambda_
            secret_binary = secret_binary[:-num_of_supplement_bits]  # 移除补位

        # 如果指定了提取位数，则只返回所需数量的位
        if self.bits_to_extract is not None:
            secret_binary = secret_binary[:self.bits_to_extract]
        print(f"Recovered secret: {secret_binary}")
        return secret_binary

class SecretComparator:
    def __init__(self, original_secret, recovered_secret):
        self.original_secret = original_secret
        self.recovered_secret = recovered_secret

    def hamming_distance(self):
        # 确保两个字符串长度相同
        length = min(len(self.original_secret), len(self.recovered_secret))
        return sum(1 for i in range(length) if self.original_secret[i] != self.recovered_secret[i])

    def similarity_percentage(self):
        length = min(len(self.original_secret), len(self.recovered_secret))
        if length == 0:
            return 0.0
        matches = length - self.hamming_distance()
        return (matches / length) * 100


def generate_secret(length):
    secret = ''
    for _ in range(length):
        bit = secrets.choice([0, 1])  # 使用 secrets 模块生成随机数
        secret += str(bit)
    return secret




def save_img_from_sample(model, samples_ddim, fname):
    x_samples_ddim = model.decode_first_stage(samples_ddim)
    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
    x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
    x_sample = 255. * rearrange(x_image_torch[0].cpu().numpy(), 'c h w -> h w c')
    img = Image.fromarray(x_sample.astype(np.uint8))
    img.save(fname)


def feat_merge(opt, cnt_feats, sty_feats, start_step=0):
    feat_maps = [{'config': {
        'gamma': opt.gamma,
        'T': opt.T,
        'timestep': _,
    }} for _ in range(50)]

    for i in range(len(feat_maps)):
        if i < (50 - start_step):
            continue
        cnt_feat = cnt_feats[i]
        sty_feat = sty_feats[i]
        ori_keys = sty_feat.keys()

        for ori_key in ori_keys:
            if ori_key[-1] == 'q':
                feat_maps[i][ori_key] = cnt_feat[ori_key]
            if ori_key[-1] == 'k' or ori_key[-1] == 'v':
                feat_maps[i][ori_key] = sty_feat[ori_key]
    return feat_maps


def load_img(path):
    image = Image.open(path).convert("RGB")
    x, y = image.size
    print(f"Loaded input image of size ({x}, {y}) from {path}")
    h = w = 512
    image = transforms.CenterCrop(min(x, y))(image)
    image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.


def adain(cnt_feat, sty_feat):
    cnt_mean = cnt_feat.mean(dim=[0, 2, 3], keepdim=True)
    cnt_std = cnt_feat.std(dim=[0, 2, 3], keepdim=True)
    sty_mean = sty_feat.mean(dim=[0, 2, 3], keepdim=True)
    sty_std = sty_feat.std(dim=[0, 2, 3], keepdim=True)
    output = ((cnt_feat - cnt_mean) / cnt_std) * sty_std + sty_mean
    return output, cnt_mean, cnt_std, sty_mean, sty_std

def reverse_adain(output, cnt_mean, cnt_std, sty_mean, sty_std):
    cnt_feat = ((output - sty_mean) / sty_std) * cnt_std + cnt_mean
    return cnt_feat


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sty', default='./data/Fauvism')
    parser.add_argument('--ddim_inv_steps', type=int, default=50, help='DDIM eta')
    parser.add_argument('--save_feat_steps', type=int, default=50, help='DDIM eta')
    parser.add_argument('--start_step', type=int, default=49, help='DDIM eta')
    parser.add_argument('--ddim_eta', type=float, default=0.0, help='DDIM eta')
    parser.add_argument('--H', type=int, default=512, help='image height, in pixel space')
    parser.add_argument('--W', type=int, default=512, help='image width, in pixel space')
    parser.add_argument('--C', type=int, default=4, help='latent channels')
    parser.add_argument('--f', type=int, default=8, help='downsampling factor')
    parser.add_argument('--T', type=float, default=1.2, help='attention temperature scaling hyperparameter')
    parser.add_argument('--gamma', type=float, default=0.5, help='query preservation hyperparameter')
    parser.add_argument("--attn_layer", type=str, default='6,7,8,9,10,11', help='injection attention feature layers')
    parser.add_argument('--model_config', type=str, default='models/ldm/stable-diffusion-v1/v1-inference.yaml',
                        help='model config')
    parser.add_argument('--precomputed', type=str, default='./precomputed_feats',
                        help='save path for precomputed feature')
    parser.add_argument('--ckpt', type=str, default='models/ldm/stable-diffusion-v1/model.ckpt',
                        help='model checkpoint')
    parser.add_argument('--precision', type=str, default='autocast', help='choices: ["full", "autocast"]')
    parser.add_argument('--output_path01', type=str, default='./output/1')
    parser.add_argument('--output_path_cnt', type=str, default='./output/1')
    parser.add_argument("--without_init_adain", action='store_true')
    parser.add_argument("--without_attn_injection", action='store_true')
    opt = parser.parse_args()

    feat_path_root = opt.precomputed
    seed_everything(1)

    output_path01 = opt.output_path01
    output_path_cnt = opt.output_path_cnt
    os.makedirs(output_path01, exist_ok=True)
    os.makedirs(output_path_cnt, exist_ok=True)
    if len(feat_path_root) > 0:
        os.makedirs(feat_path_root, exist_ok=True)

    model_config = OmegaConf.load(f"{opt.model_config}")
    model = load_model_from_config(model_config, f"{opt.ckpt}")

    self_attn_output_block_indices = list(map(int, opt.attn_layer.split(',')))
    ddim_inversion_steps = opt.ddim_inv_steps
    save_feature_timesteps = ddim_steps = opt.save_feat_steps

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    unet_model = model.model.diffusion_model
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)
    time_range = np.flip(sampler.ddim_timesteps)
    idx_time_dict = {}
    time_idx_dict = {}
    for i, t in enumerate(time_range):
        idx_time_dict[t] = i
        time_idx_dict[i] = t

    seed = torch.initial_seed()
    opt.seed = seed

    global feat_maps
    feat_maps = [{'config': {
        'gamma': opt.gamma,
        'T': opt.T
    }} for _ in range(50)]

    def ddim_sampler_callback(pred_x0, xt, i):
        save_feature_maps_callback(i)
        save_feature_map(xt, 'z_enc', i)

    def save_feature_maps(blocks, i, feature_type="input_block"):
        block_idx = 0
        for block_idx, block in enumerate(blocks):
            if len(block) > 1 and "SpatialTransformer" in str(type(block[1])):
                if block_idx in self_attn_output_block_indices:
                    # self-attn
                    q = block[1].transformer_blocks[0].attn1.q
                    k = block[1].transformer_blocks[0].attn1.k
                    v = block[1].transformer_blocks[0].attn1.v
                    save_feature_map(q, f"{feature_type}_{block_idx}_self_attn_q", i)
                    save_feature_map(k, f"{feature_type}_{block_idx}_self_attn_k", i)
                    save_feature_map(v, f"{feature_type}_{block_idx}_self_attn_v", i)
            block_idx += 1

    def save_feature_maps_callback(i):
        save_feature_maps(unet_model.output_blocks, i, "output_block")

    def save_feature_map(feature_map, filename, time):
        global feat_maps
        cur_idx = idx_time_dict[time]
        feat_maps[cur_idx][f"{filename}"] = feature_map

    start_step = opt.start_step
    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    uc = model.get_learned_conditioning([""])
    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
    sty_img_list = sorted(os.listdir(opt.sty))


    begin = time.time()
    # start_index = 286  # 从第644个图片开始（索引从0开始）
    #
    # for index, sty_name in enumerate(sty_img_list):
    #     if index< start_index:
    #         continue  # 跳过前面的图片
    for sty_name in sty_img_list:

        sty_name_ = os.path.join(opt.sty, sty_name)
        init_sty = load_img(sty_name_).to(device)
        seed = -1
        sty_feat_name = os.path.join(feat_path_root, os.path.basename(sty_name).split('.')[0] + '_sty.pkl')
        sty_z_enc = None

        if len(feat_path_root) > 0 and os.path.isfile(sty_feat_name):
            print("Precomputed style feature loading: ", sty_feat_name)
            with open(sty_feat_name, 'rb') as h:
                sty_feat = pickle.load(h)
                sty_z_enc = torch.clone(sty_feat[0]['z_enc'])
        else:
            init_sty = model.get_first_stage_encoding(model.encode_first_stage(init_sty))
            sty_z_enc, _ = sampler.encode_ddim(init_sty.clone(), num_steps=ddim_inversion_steps,
                                               unconditional_conditioning=uc, \
                                               end_step=time_idx_dict[ddim_inversion_steps - 1 - start_step], \
                                               callback_ddim_timesteps=save_feature_timesteps,
                                               img_callback=ddim_sampler_callback)
            sty_feat = copy.deepcopy(feat_maps)
            sty_z_enc = feat_maps[0]['z_enc']


        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    # secret
                    length = 64
                    # for _ in range(286):
                    #     secret = generate_secret(length)
                    secret = generate_secret(length)

                    lambda_ = 1
                    nz = 4 * 64 * 64  # 噪声向量的长度

                    secret_to_tensor = SecretToTensor(secret, lambda_, nz)
                    target_shape = (1, 4, 64, 64)
                    noise_tensor, special_positions, secret, b_index= secret_to_tensor.secret_mapping()  # 获取 special_positions

                    # 获取当前文件夹中所有文件的列表
                    existing_files = os.listdir(output_path_cnt)

                    # 过滤出所有与图像文件相关的文件（假设为 png 格式）
                    existing_files = [f for f in existing_files if
                                      f.startswith("recover_content") and f.endswith(".png")]

                    # 计算下一个文件编号
                    file_counter = len(existing_files) + 1

                    # 创建新的文件名
                    output_name = f"recover_content_{file_counter}.png"
                    # inversion
                    # 确保所有输入张量都在同一设备上
                    noise_tensor = torch.tensor(noise_tensor, dtype=torch.float32).view(target_shape).to(device)
                    print(f"Inversion end: {time.time() - begin}")
                    length_ = len(secret)
                    print(f"Length of secret: {length_}")
                    print(f" secret: {secret}")

                    # inference
                    samples_ddim, intermediates = sampler.sample(S=ddim_steps,
                                                                 batch_size=1,
                                                                 shape=shape,
                                                                 verbose=False,
                                                                 unconditional_conditioning=uc,
                                                                 eta=opt.ddim_eta,
                                                                 x_T=noise_tensor,
                                                                 injected_features=None,
                                                                 start_step=start_step,
                                                                 )

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                    x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
                    x_sample = 255. * rearrange(x_image_torch[0].cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))

                    img.save(os.path.join(output_path_cnt, output_name))


                    # 加载生成的图片并进行加噪
                    cnt_name_ = os.path.join(output_path_cnt, output_name)
                    init_cnt = load_img(cnt_name_).to(device)
                    cnt_feat = None
                    # ddim inversion encoding
                    init_cnt = model.get_first_stage_encoding(model.encode_first_stage(init_cnt))
                    cnt_z_enc, _ = sampler.encode_ddim(init_cnt.clone(), num_steps=ddim_inversion_steps,
                                                       unconditional_conditioning=uc,
                                                       end_step=time_idx_dict[ddim_inversion_steps - 1 - start_step],
                                                       callback_ddim_timesteps=save_feature_timesteps,
                                                       img_callback=ddim_sampler_callback)
                    cnt_feat = copy.deepcopy(feat_maps)
                    cnt_z_enc = noise_tensor

                    cnt_name = output_name
                    # inversion
                    output_name = f"{os.path.basename(cnt_name).split('.')[0]}_stylized_{os.path.basename(sty_name).split('.')[0]}.png"

                    print(f"Inversion end: {time.time() - begin}")
                    if opt.without_init_adain:
                        adain_z_enc = cnt_z_enc
                    else:
                        adain_z_enc, cnt_mean, cnt_std, sty_mean, sty_std = adain(cnt_z_enc, sty_z_enc)
                    feat_maps = feat_merge(opt, cnt_feat, sty_feat, start_step=start_step)
                    if opt.without_attn_injection:
                        feat_maps = None

                    # inference
                    samples_ddim, intermediates = sampler.sample(S=ddim_steps,
                                                                 batch_size=1,
                                                                 shape=shape,
                                                                 verbose=False,
                                                                 unconditional_conditioning=uc,
                                                                 eta=opt.ddim_eta,
                                                                 x_T=adain_z_enc,
                                                                 injected_features=feat_maps,
                                                                 start_step=start_step,
                                                                 )

                    # final_ddim_tensor = samples_ddim.clone()
                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                    x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
                    x_sample = 255. * rearrange(x_image_torch[0].cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))

                    img.save(os.path.join(output_path01, output_name))

                    if len(feat_path_root) > 0:
                        print("Save features")
                        if not os.path.isfile(sty_feat_name):
                            with open(sty_feat_name, 'wb') as h:
                                pickle.dump(sty_feat, h)


                    feat_maps_reversed = feat_maps[::-1]

                    begin = time.time()


                    output_name_ = os.path.join(opt.output_path01, output_name)
                    init_output = load_img(output_name_).to(device)
                    seed = -1
                    output_z_enc = None

                    # init_output = final_ddim_tensor.clone()  # 确保使用 final_ddim_tensor
                    init_output = model.get_first_stage_encoding(model.encode_first_stage(init_output))
                    output_z_enc, _ = sampler.reverse_sample(
                        init_output.clone(),
                        num_steps=ddim_inversion_steps,
                        unconditional_conditioning=uc,
                        end_step=time_idx_dict[ddim_inversion_steps - 1 - start_step],
                        callback_ddim_timesteps=save_feature_timesteps,
                        injected_features=feat_maps_reversed,
                        img_callback=ddim_sampler_callback
                    )
                    output_z_enc = feat_maps[0]['z_enc']

                    cnt_feat = reverse_adain(output_z_enc, cnt_mean, cnt_std, sty_mean, sty_std)

                    # cnt_feat_np = cnt_feat.detach().cpu().numpy()
                    # np.savetxt('cnt_feat.txt', cnt_feat_np.flatten(), fmt='%.6f')

                    reverse_noise_tensor = cnt_feat
                    length = b_index
                    tensor_to_secret = TensorToSecret(reverse_noise_tensor, lambda_, length)
                    recovered_secret = tensor_to_secret.tensor_to_secret(special_positions)

                    recovered_length = len(recovered_secret)
                    print(f"Length of recovered_secret: {recovered_length}")

                    comparator = SecretComparator(secret, recovered_secret)
                    hamming_dist = comparator.hamming_distance()
                    similarity = comparator.similarity_percentage()

                    print(f"Hamming Distance: {hamming_dist}")
                    print(f"Similarity Percentage: {similarity:.2f}%")

                    # 定义日志文件路径
                    log_file_path = 'new_similarity_log64-T-1.2-0.5.txt'

                    # 将每次的相似度值写入日志文件
                    with open(log_file_path, 'a') as log_file:
                        log_file.write(f"{similarity:.2f}\n")

    # 在程序结束时计算所有相似度的平均值
    def calculate_average_similarity(log_file_path):
        if not os.path.exists(log_file_path):
            return 0.0  # 如果日志文件不存在，返回0

        with open(log_file_path, 'r') as log_file:
            similarities = [float(line.strip()) for line in log_file.readlines()]

        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            return avg_similarity
        return 0.0  # 如果没有相似度数据，则返回0

    # # 计算平均相似度
    log_file_path = 'new_similarity_log64-T-1.2-0.5.txt'

    # 在程序结束时计算所有相似度的平均值
    avg_similarity = calculate_average_similarity(log_file_path)

    # 输出平均相似度
    print(f"Average Similarity Percentage: {avg_similarity:.2f}%")

    # 将平均相似度写入日志文件
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"Average Similarity Percentage: {avg_similarity:.2f}%\n")

if __name__ == "__main__":
    main()
