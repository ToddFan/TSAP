import os
from random import random, sample

import torch

from utils_patch import PatchApplier
from ptop import ParticleToPatch


class OptimizeFunction:
    def __init__(self, detector, patch_size, device, dimension):
        self.detector = detector

        self.device = device
        self.ptp = ParticleToPatch(dimension)
        self.pa = PatchApplier()
        self.size = 0
        self.num_patch = 1
        self.patch_size = patch_size

    def set_para(self, targets, imgs, segs):
        self.segs = segs
        self.targets = targets
        self.imgs = imgs

    def evaluate(self, x):
        # x: 5列：(x,y,c,l,α)
        with torch.no_grad():
            patch_tf, patch_mask_tf = self.ptp(x, self.targets, self.imgs)
            imgWithPatch = self.pa(self.imgs, self.segs, patch_tf, patch_mask_tf)

            yolov5_out, yolov5_train_out = self.detector(imgWithPatch)
            yolov5_obj_confidence = yolov5_out[:, :, 4]

            # 按第二维度降序对 obj_confidence 进行排序
            yolov5_sorted_obj_confidence, _ = torch.sort(yolov5_obj_confidence, dim=1, descending=True)
            # 从排序后的张量中选择前1000个值
            yolov5_top_1000_confidence = yolov5_sorted_obj_confidence[:, :1000]
            # 计算前1000个值的总和
            yolov5_sum_obj_confidence = torch.sum(yolov5_top_1000_confidence, dim=1)

            yolov5_obj_loss = torch.mean(yolov5_sum_obj_confidence)

            return_obj_loss = yolov5_obj_loss

        return return_obj_loss


class Theta:
    def __init__(self, blocks_num, device):
        self.device = device
        self.blocks_num = blocks_num
        self.bounds = torch.tensor([(0, 0.88), (0, 0.88), (0, 1), (25, 25), (0, 180)], device=device)

        positions = torch.rand(blocks_num, 2).to(device)

        pix_val = torch.rand(blocks_num, 1).to(device)

        blocks_l = torch.tensor([25.0] * blocks_num, dtype=torch.float, device=device).unsqueeze(1)

        alpha = (torch.rand(blocks_num, 1) * 180.0).to(device)

        self.attr = ensure_bounds(torch.cat([positions, pix_val, blocks_l, alpha], dim=1), self.bounds)


class DE:
    def __init__(self, popsize, device):
        self.best_sol = r"runs/result/best_solution.txt"
        self.max_iterations = 10
        self.popsize = popsize
        self.bounds = torch.tensor([(0, 0.88), (0, 0.88), (0, 1), (25, 25), (0, 180)], device=device)
        self.mutate = 0.5
        self.recombination = 0.6
        self.best_score = None
        self.best_theta = None
        self.gen_scores = None

        self.population = []
        for i in range(0, popsize):  # popsize 种群向量数量。
            new_theta = Theta(blocks_num=7, device=device)
            indv = new_theta.attr
            self.population.append(indv)  # 将向量加入到种群

    def optimize(self, function):
        self.fitness_function = function

    def run(self):

        # --- Run
        # for iteration in range(self.max_iterations):
        from tqdm import trange

        if self.best_score == None:
            self.gen_scores = []
            for i in range(0, self.popsize):
                x = self.population[i].cuda()
                score = self.fitness_function.evaluate(x)
                self.gen_scores.append(score)
            self.best_score = min(self.gen_scores)
            self.best_theta = self.population[self.gen_scores.index(min(self.gen_scores))]

        for iteration in trange(self.max_iterations, desc='     DE----->'):  # 使用tqdm的trange函数替代range函数

            # 自适应的差分算法
            import math
            mutate = self.mutate * (2 ** (math.e ** (1 - self.max_iterations / (self.max_iterations - iteration))))

            # cycle through each individual in the population 遍历种群中所有变量
            for j in range(0, self.popsize):

                # --- MUTATION (step #3.A) ---------------------+ 变异，生成新的向量参数

                # select three random vector index positions [0, popsize), not including current vector (j) 种群随机选取三个向量，不包含当前向量
                candidates = list(range(0, self.popsize))
                candidates.remove(j)
                random_index = sample(candidates, 3)  # 从candidates中随机选三个

                x_1 = self.population[random_index[0]].cuda()
                x_2 = self.population[random_index[1]].cuda()
                x_3 = self.population[random_index[2]].cuda()
                x_t = self.population[j]  # target individual

                # subtract x3 from x2, and create a new vector (x_diff) 【x_2 - x_3】
                x_diff = x_2 - x_3

                # multiply x_diff by the mutation factor (F) and add to x_1  【x_diff * 变异因子 + x_1】

                # v_donor = x_1 + self.mutate * x_diff  # 原de
                v_donor = x_1 + mutate * x_diff  # 自适应的差分算法
                v_donor = ensure_bounds(v_donor, self.bounds)  # 确保在范围内 v_donor

                # --- RECOMBINATION (step #3.B) ----------------+  与当前向量进行参数混合生成实验向量

                v_trial = []
                x_t = x_t.tolist()
                v_donor = v_donor.tolist()
                for k in range(len(x_t)):
                    crossover = random()
                    if crossover <= self.recombination:
                        v_trial.append(v_donor[k])

                    else:
                        v_trial.append(x_t[k])

                # --- GREEDY SELECTION (step #3.C) -------------+ 贪心选择
                v_trial = torch.tensor(v_trial).cuda()
                x_t = torch.tensor(x_t).cuda()

                score_trial = self.fitness_function.evaluate(v_trial)  # 实验向量的分数
                # score_target = self.fitness_function.evaluate(x_t)  # 目标向量的分数
                score_target = self.gen_scores[j]  # 目标向量的分数

                if score_trial < score_target:  # 目标向量分数较高，保存实验分数（取分数低的）
                    self.population[j] = v_trial
                    self.gen_scores[j] = score_trial
                    # print('   >', score_trial, v_trial)

                # else:
                # print('   >', score_target, x_t)
                # self.gen_scores.append(score_target)

            # --- SCORE KEEPING --------------------------------+

            # gen_avg = sum(gen_scores) / self.popsize  # current generation avg. fitness
            gen_best = min(self.gen_scores)  # fitness of best individual
            gen_sol = self.population[self.gen_scores.index(min(self.gen_scores))]  # solution of best individual

            if gen_best < self.best_score:
                self.best_score = gen_best
                self.best_theta = gen_sol

            # print('      > GENERATION AVERAGE:', gen_avg)
            print('      > GENERATION BEST:', self.best_score)
            print('         > BEST SOLUTION:', self.best_theta, '\n')

            if iteration == self.max_iterations - 1:
                save_path = os.path.dirname(self.best_sol)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                with open(self.best_sol, 'w') as f:
                    f.write("best_solution:\n")
                    f.write(f"{self.best_theta}\n")
                    f.write("best_score:\n")
                    f.write(f"{self.best_score}\n")

        pass


def ensure_bounds(vec, bounds):  # 确保向量在可选范围内

    import random

    for i in range(len(vec[0])):
        for j in range(len(vec)):
            if vec[j][i] < bounds[i][0] or vec[j][i] > bounds[i][1]:
                vec[j][i] = random.uniform(bounds[i][0], bounds[i][1])

    return vec
