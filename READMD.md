{"nbformat":4,"nbformat_minor":0,"metadata":{"colab":{"provenance":[]},"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python"}},"cells":[{"cell_type":"markdown","source":["1. Each model can be directly run with corresponding ipynb. Unfortunately you do need to install all dependencies every single time. Remember to manually selcect python 3.6 in the second box after it's installed!\n"],"metadata":{"id":"IeuUTWwFMHG6"}},{"cell_type":"markdown","source":["2. In the project folder, I made a shortcut to the data files you shared with me and hopefully you will be able to access without problem.  "],"metadata":{"id":"5OPXaIKQMlFc"}},{"cell_type":"markdown","source":["3. Results can be found in /TPC-LoS-prediction/models/experiments/final/eICU/LOS"],"metadata":{"id":"gqoZPyJcMwrH"}},{"cell_type":"markdown","source":["4. I ultimately bought ColabPro+ which allows background execution, which is the key as I realized. "],"metadata":{"id":"-CRh0_RiNYLZ"}},{"cell_type":"markdown","source":["5. With the exception of the two CW-LSTM model, I used standard GPU to run all other models without problem. For CW-LSTM, I had to use premium GPU support with NAVIDA A100 due to signifnicant GPU usage. However to do this, I had to update pytorch/CUDA version as a seprate step in these two models. Fortunately I did not see any obvious compatibility issues. "],"metadata":{"id":"mi6UIRxPNkWA"}},{"cell_type":"markdown","source":["6. There are several manual changes I made to original codes in the process of debuging, including: \n","\n","\n","*   I got this error when running any model in test mode:\n","\n","`Expected more than 1 value per channel when training, got input size torch.Size([1, 64]) Experiment exited. Checkpoints stored =)`\n","\n","I think this is due to the last batch of test set has a size of 1. To debug the issue I added this line in experiment_template.py on line 254:\n","\n","```\n","if batch[-1].size(dim=0) == 1: \n","    continue\n","```\n","\n","\n","\n","\n","*   I got this error when running temp_only model:\n","\n","`File \"/content/gdrive/MyDrive/TPC-Project/TPC-LoS-prediction/models/tpc_model.py\", line 569, in forward next_X = torch.stack(X_separated, dim=2).view(B, 2 * self.F, T) # B * 2F * T RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead. Experiment exited. Checkpoints stored =)`\n","\n","To fix this, in line 570 of tpc_model.py, I added contiguous() infont of the view\n"],"metadata":{"id":"8Hvs9brTOInj"}},{"cell_type":"markdown","source":["7. Of note, the way I installed requirements using pip did not install shap package correctly. Though this does not seem to matter as it is only used in our analysis."],"metadata":{"id":"mPMaekdePsK5"}},{"cell_type":"markdown","source":["8. I wrote separate pythons files to run all models using 50% of train data. These can be found in /TPC-LoS-prediction/models/final_experiment_scripts/eICU/LoS/. You likely would need to do the same to run new experiments."],"metadata":{"id":"3KtgcR6zQ9Jm"}},{"cell_type":"markdown","source":["9. To run new experiment, not sure if you can just write a new ipynb in my shared drive. I shared google drive with you as an editor."],"metadata":{"id":"O0H5gl08RrKH"}},{"cell_type":"markdown","source":["10. I have updated the project draft in Overleaf. "],"metadata":{"id":"uKtvddgeRsyU"}}]}