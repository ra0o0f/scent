import torch
import mlflow

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.max = float('-inf')
        self.min = float('inf')
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        try:
            self.max = max(val, self.max)
            self.min = min(val, self.min)
        except Exception:
            pass
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class TrainLog:

    def __init__(self, rank = 0, disabled = False, project=None, batch_log=True, log_step=None):
        self.rank = rank
        self.disabled = disabled
        self.project = project
        self.batch_log = batch_log
        self.log_step = log_step

        if self.batch_log:
            self.metrics = {}

        if not self.disabled:
            mlflow.end_run()
            mlflow.set_experiment(project)
            mlflow.start_run()
            self.mlflow_run = mlflow.active_run()

        self.avg_meters = {}

    def end(self):
        if not self.disabled:
            mlflow.end_run()

    def start_step(self, global_step):
        if self.disabled:
            return
        
        self.global_step = global_step
        
        if self.batch_log:
            self.metrics = {}

    def end_step(self):
        if self.disabled:
            return
        
        if (self.global_step % self.log_step)!=0:
            return
        
        if self.batch_log:
            mlflow.log_metrics(self.metrics, step=self.global_step)

    def track(self,
              value,
              name: str = None,
              force_step = None):
        
        if self.disabled:
            return
        
        if force_step:
            mlflow.log_metric(key=name, value=value, step=force_step)
            return
        
        if (self.global_step % self.log_step)!=0:
            return

        with torch.no_grad():
            if self.batch_log:
                self.metrics[name] = value
            # else:
            #     mlflow.log_metric(key=name, value=value, step=self.global_step)


    def add_averagemeter(self, name, val, n=1, print_msg = None):

        if name not in self.avg_meters:
            self.avg_meters[name] = AverageMeter()

        am:AverageMeter = self.avg_meters[name]

        self.track(val, name=name)
        am.update(val,n=n)
        self.track(am.avg, name=f"{name}_avg")

        if print_msg is not None:
            print_msg.append(f',{name}: {am.avg:.4f}, ') 

        return am

    def add_output_stats(self, name, o, add_extremum=False):

        if self.disabled:
            return
        
        if self.rank != 0:
            return

        with torch.no_grad():

            o_mean = o.mean().item()
            o_std = o.std().item()
            o_norm = o.norm().item()

            self.track(o_mean, name=f'{name}/mean')
            self.track(o_std, name=f'{name}/std')
            self.track(o_norm, name=f'{name}/norm')

            if add_extremum:
                o_min = o.min().item()
                o_max = o.max().item()
                self.track(o_min, name=f'{name}/min')
                self.track(o_max, name=f'{name}/max')

    def track_grad(self, grad_logger, named_params, context):

        if self.disabled:
            return
        
        if self.rank != 0:
            return

        for n, p in named_params:
            if (p.grad is not None) and not (n.endswith('.bias') or len(p.shape) == 1):

                ctx = context.copy()
                if not grad_logger(n, ctx):
                     continue
                    
                grad_min = p.grad.min().item()
                grad_max = p.grad.max().item()
                grad_mean = p.grad.mean().item()
                grad_std = p.grad.std().item()
                grad_norm = p.grad.norm().item()
                grad_std_ratio = (p.grad.std() / p.std()).item()

                self.track(grad_min, name=f'grad-{n}/min')

                self.track(grad_max, name=f'grad-{n}/max')

                self.track(grad_mean, name=f'grad-{n}/mean')

                self.track(grad_std, name=f'grad-{n}/std')

                self.track(grad_norm, name=f'grad-{n}/norm')

                self.track(grad_std_ratio, name=f'grad-{n}std-ratio')
