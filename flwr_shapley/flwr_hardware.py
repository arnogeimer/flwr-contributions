import torchprofile as tp

# macs = tp.profile_macs(conv_model,(x,))
# flops = 2*macs

''' 
Size

model = models.resnet18()
param_size = 0
for param in model.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2
print('model size: {:.3f}MB'.format(size_all_mb))
'''

'''
TFLOPS = (Number of CUDA Cores * Clock Speed * Operations per Core per Cycle) / 1,000,000,000,000. 
'''

# cuda cores != tensor cores
# assume only cuda cores for the beginning

'''NeuSight — “Forecasting GPU Performance for Deep Learning Training and Inference” (2024) — framework to predict performance on unseen GPUs without executing target model on them (models hardware & software optimizations).
arXiv

Good for: predicting runtime across hardware and unseen setups.

ETS — “Deep Learning Training Iteration Time Prediction based on Execution Trace Sliding Windows” (ACM, 2024) — iteration-time predictor using execution traces.
ACM Digital Library

PreNeT — “Leveraging Computational Features to Predict Deep Neural [Training Duration]” (ACM, 2025-ish) — claims accurate prediction across unseen hardware infrastructures; focuses on computational features.
ACM Digital Library

DNNPerf — “Runtime Performance Prediction for Deep Learning Models” (Microsoft Research, 2021, PDF) — predicts runtime, memory consumption, and analyses of framework/library effects. Useful systems paper with practical models.
Microsoft

“AI and the Memory Wall” (arXiv, 2024) — analysis showing memory bandwidth becoming the dominant limiter for some model classes (esp. decoder transformers); explores implications for architecture & training.
arXiv

Epoch.ai blog — “Estimating Training Compute of Deep Learning Models” (2022) — practical walkthrough of estimating training compute by counting ops and mapping to GPU time; good primer.
Epoch AI

“Predicting Model Training Time to Optimize Distributed Machine Learning Applications” (conference/tech report, ~2021) — older but specifically about predicting training time to plan scheduling.
ResearchGate

Survey: “Deep Learning Workload Scheduling in GPU Datacenters” (ACM survey) — covers scheduling and performance factors relevant to runtime estimation in multi-tenant/cloud environments. 
'''

'''https://github.com/sitar-lab/NeuSight'''

# https://www.deepspeed.ai/tutorials/flops-profiler/
# github.com/facebookresearch/fvcore/blob/main/docs/flop_count.md

# set_per_process_memory_fraction

