import pandas as pd
import numpy as np
from Bio import Phylo
from scipy.optimize import minimize
from statsmodels.stats.weightstats import DescrStatsW



import sys
sys.path.append('../../Tree_weighting/Code/')
import weighting_methods

#####This is making the weights as a dataframe in an attempted speed-up...

def ml_root_weighted(tree):
    ###Depths are important! This is what I am trying to optimize in terms
    ###of making these look as close to normal as possible. So this gets the
    ###starting depths as a DataFrame and subsequent tree crawling adds/subtracts
    ###to these values
    initial_depths = tree.root.depths()
    terminal_depths_df = pd.DataFrame()
    terminal_depths_df['depth'] = np.nan
    for term in tree.get_terminals():
        terminal_depths_df.set_value(term.name, 'depth', initial_depths[term])
    depths_dict = {}
    depths_dict[tree.root] = terminal_depths_df
    
    ###Getting starting weights
    weights_dict_single = weighting_methods.GSC_adhock_extended(tree)
    weights_df = pd.DataFrame(list(weights_dict_single.values()),
                              index=[term.name for term in weights_dict_single.keys()])
    weights_dict_all = {}
    weights_dict_all[tree.root] = weights_df

    explored, function_optima, depths_dict, weights_dict =\
            recursive_crawl_ml(tree.root, [], [], depths_dict, weights_dict_all, tree)
    
    ###Getting the best function eval and rooting there
    function_optima = sorted(function_optima, key=lambda x: x[1].fun)
    tree.root_with_outgroup(function_optima[0][0], outgroup_branch_length=0.)
    assert tree.root.clades[1].branch_length == 0.
    assert tree.root.clades[1] == function_optima[0][0]
    tree.root.clades[0].branch_length -= function_optima[0][1].x[0]
    tree.root.clades[1].branch_length += function_optima[0][1].x[0]
    return tree, function_optima, depths_dict, weights_dict

def recursive_crawl_ml(hypothetical_root, explored, function_optima, depths_dict, weights_dict, tree):
    if len(hypothetical_root.clades) == 2:
        l_clade, r_clade = hypothetical_root.clades
        l_bl = l_clade.branch_length
        r_bl = r_clade.branch_length
        #L clade first
        if l_bl > 0:
            depths_dict, downstream_terms, upstream_terms =\
                    update_depth_df_dict(depths_dict, l_clade, hypothetical_root)
            weights_dict =\
                    update_weights_dict(weights_dict, l_clade, hypothetical_root, downstream_terms, upstream_terms)
            res = optimize_root_loc_on_branch(l_clade, depths_dict[l_clade], weights_dict[l_clade], downstream_terms, upstream_terms)
            function_optima.append((l_clade, res))
            explored, function_optima, depths_dict, weights_dict =\
                    recursive_crawl_ml(l_clade, explored, function_optima, depths_dict, weights_dict, tree)
        #R clade second
        if r_bl > 0:
            depths_dict, downstream_terms, upstream_terms =\
                    update_depth_df_dict(depths_dict, r_clade, hypothetical_root)
            weights_dict =\
                    update_weights_dict(weights_dict, r_clade, hypothetical_root, downstream_terms, upstream_terms)
            res = optimize_root_loc_on_branch(r_clade, depths_dict[r_clade], weights_dict[r_clade], downstream_terms, upstream_terms)
            function_optima.append((r_clade, res))
            explored, function_optima, depths_dict, weights_dict =\
                    recursive_crawl_ml(r_clade, explored, function_optima, depths_dict, weights_dict, tree)
    elif len(hypothetical_root.clades) == 0:
        explored.append(hypothetical_root)
        return explored, function_optima, depths_dict, weights_dict
    
    else:
        print('Some big error here with the number of clades stemming from this root')
    explored.append(hypothetical_root)
    return explored, function_optima, depths_dict, weights_dict

def update_depth_df_dict(depths_dict, my_clade, parent_clade):
    downstream_terms = [i.name for i in my_clade.get_terminals()]
    upstream_terms = list(set(list(depths_dict[parent_clade].index)) - set(downstream_terms))
    depths_dict[my_clade] = depths_dict[parent_clade].copy(deep=True)
    depths_dict[my_clade].loc[downstream_terms, 'depth'] -= my_clade.branch_length
    depths_dict[my_clade].loc[upstream_terms, 'depth'] += my_clade.branch_length
    return depths_dict, downstream_terms, upstream_terms

def update_weights_dict(weights_dict, my_clade, parent_clade, downstream_terms, upstream_terms):
    '''
    Some convoluted copy things happening here that should be double checked
    '''
    weights_dict[my_clade] = weights_dict[parent_clade].reindex(downstream_terms+upstream_terms)
    end_col = weights_dict[my_clade].columns[-1]
    weights_dict[my_clade].loc[downstream_terms] =\
            np.insert(weights_dict[my_clade].loc[downstream_terms].values[:,:-1], 0, 0, axis=1)
    bl_to_disperse = my_clade.branch_length
    to_divide = weights_dict[my_clade].loc[upstream_terms, end_col].sum()
    to_add = weights_dict[my_clade].loc[upstream_terms, end_col] / to_divide * bl_to_disperse
    new_col = weights_dict[my_clade].loc[upstream_terms, end_col] + to_add
    weights_dict[my_clade].loc[upstream_terms] =\
            np.append(weights_dict[my_clade].loc[upstream_terms].values,\
                      new_col.values.reshape(-1, 1), axis=1)[:,1:]   
    return weights_dict

def optimize_root_loc_on_branch(my_clade, depths_df, weights_df, downstream_terms, upstream_terms):
    '''
    '''    
    downstream_dists = np.array(depths_df.loc[downstream_terms, 'depth'])
    upstream_dists = np.array(depths_df.loc[upstream_terms, 'depth'])

    end_col = weights_df.columns[-1]
    second_to_last_col = weights_df.columns[-2]
    downstream_weights = np.array(weights_df.loc[downstream_terms, end_col])
    upstream_weights = np.array(weights_df.loc[upstream_terms, end_col])
    old_upstream_weights = np.array(weights_df.loc[upstream_terms, second_to_last_col])

    bl_bounds = np.array([[0., my_clade.branch_length]])
    ###Valid options for method are L-BFGS-B, SLSQP and TNC
    res = minimize(branch_scan_ml, np.array(np.mean(bl_bounds)),\
                          args=(downstream_dists, upstream_dists,\
                                downstream_weights, upstream_weights, old_upstream_weights),\
                          bounds=bl_bounds, method='SLSQP')
    return res

def branch_scan_ml(modifier, ds_dists, us_dists, ds_weights, us_weights, old_us_weights):
    temp_ds_dists = ds_dists + modifier
    temp_us_dists = us_dists - modifier
    all_dists = np.concatenate((temp_ds_dists, temp_us_dists))
    
    total_ds = np.sum(ds_weights)
    if total_ds != 0:
        temp_ds_weights = ds_weights + (ds_weights/total_ds*modifier)
    else:
        temp_ds_weights = ds_weights + modifier


    total_us = np.sum(old_us_weights)
    if total_us != 0:
        temp_us_weights = us_weights - (old_us_weights/total_us*modifier)
    else:
        temp_us_weights = us_weights - modifier
    all_weights = np.concatenate((temp_ds_weights, temp_us_weights))
    dsw = DescrStatsW(all_dists, all_weights)
    return dsw.std

