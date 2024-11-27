import nnvi.eval_ray_run as eval_r
from pathlib import Path
import json
import pandas as pd

ray_folder = Path("/home/akister/ray_results")
run_name = "fn_2024-05-02_07-24-51"# Good run: "fn_2024-04-29_07-33-50"#"fn_2024-04-25_15-38-17"

results = eval_r.ray_results(ray_folder,run_name,max_len=3)
first_exp_name = list(results.ray_experiments.keys())[0]
first_exp = results.ray_experiments[first_exp_name]

m_start=first_exp.get_check_hist()[0]['chpt'].get_soln()


m = first_exp.cfg.get_soln_model()
m_v_s = first_exp.get_soln()
m_v_t = first_exp.get_test()
example = first_exp.get_example()
grid = first_exp.get_grid()
r_grid = first_exp.get_regular_grid()
loss = first_exp.get_loss()

loss_r_s = results.loss_r_matrix(empty_cash=True)
data_set = pd.DataFrame(loss_r_s)
worst_case_test=data_set.groupby(['sol_id']).loss_r.agg('max').reset_index().set_index('sol_id')
best_case_test=data_set.groupby(['sol_id']).loss_r.agg('min').reset_index().set_index('sol_id')
# worst_case_test.idxmin()
f_e_s = results.get_final_errors(metric='best_l2')#'current_l2'
f_e_d_s = pd.DataFrame([{'sol_id':k, 'l2':v} for k,v in f_e_s.items()]).set_index('sol_id')
compare_l2_to_worst=worst_case_test.join(f_e_d_s).sort_values(by='loss_r')
compare_l2_to_best=best_case_test.join(f_e_d_s).sort_values(by='loss_r')



www = eval_r.plot_data_for_nn(m_v_s,grid,pdim=results.get_dim())
hist=first_exp.get_check_hist()
i_d=eval_r.plot_data_for_nn(m_start,r_grid,pdim=results.get_dim())
#i_d[['X_x','X_y','value']]1
breakpoint()
www = first_exp.get_training_process()
df = www[["epoch","loss_r"]]
idx=(www["loss_r"]>-25 )* (www["loss_r"]<25)
breakpoint()
