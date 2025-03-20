from datetime import datetime
import yaml
import numpy as np
import warnings
import time
import pandas as pd

warnings.filterwarnings('ignore', category=yaml.YAMLLoadWarning)
pd.set_option('mode.chained_assignment', None)


### CONSTANTS
junk_cols = ['app_id']
#common_cols will be dropped when merging different sampler data to prevent the duplication
common_cols = ['timestamp', 'component_id', 'job_id']
#excluded_cols will be used to determine columns excluded from adding the sampler name, e.g., this won't be generated job_id::meminfo
excluded_cols = ['timestamp', 'unix_timestamp', 'job_id', 'component_id']
eclipse_meminfo_col_names = "timestamp,component_id,job_id,app_id,MemTotal,MemFree,MemAvailable,Buffers,Cached,SwapCached,Active,Inactive,Active(anon),Inactive(anon),Active(file),Inactive(file),Unevictable,Mlocked,SwapTotal,SwapFree,Dirty,Writeback,AnonPages,Mapped,Shmem,Slab,SReclaimable,SUnreclaim,KernelStack,PageTables,NFS_Unstable,Bounce,WritebackTmp,CommitLimit,Committed_AS,VmallocTotal,VmallocUsed,VmallocChunk,Percpu,HardwareCorrupted,AnonHugePages,CmaTotal,CmaFree,HugePages_Total,HugePages_Free,HugePages_Rsvd,HugePages_Surp,Hugepagesize,DirectMap4k,DirectMap2M,DirectMap1G".split(',')
eclipse_vmstat_col_names = "timestamp,component_id,job_id,app_id,nr_free_pages,nr_alloc_batch,nr_inactive_anon,nr_active_anon,nr_inactive_file,nr_active_file,nr_unevictable,nr_mlock,nr_anon_pages,nr_mapped,nr_file_pages,nr_dirty,nr_writeback,nr_slab_reclaimable,nr_slab_unreclaimable,nr_page_table_pages,nr_kernel_stack,nr_unstable,nr_bounce,nr_vmscan_write,nr_vmscan_immediate_reclaim,nr_writeback_temp,nr_isolated_anon,nr_isolated_file,nr_shmem,nr_dirtied,nr_written,numa_hit,numa_miss,numa_foreign,numa_interleave,numa_local,numa_other,workingset_refault,workingset_activate,workingset_nodereclaim,nr_anon_transparent_hugepages,nr_free_cma,nr_dirty_threshold,nr_dirty_background_threshold,pgpgin,pgpgout,pswpin,pswpout,pgalloc_dma,pgalloc_dma32,pgalloc_normal,pgalloc_movable,pgfree,pgactivate,pgdeactivate,pgfault,pgmajfault,pglazyfreed,pgrefill_dma,pgrefill_dma32,pgrefill_normal,pgrefill_movable,pgsteal_kswapd_dma,pgsteal_kswapd_dma32,pgsteal_kswapd_normal,pgsteal_kswapd_movable,pgsteal_direct_dma,pgsteal_direct_dma32,pgsteal_direct_normal,pgsteal_direct_movable,pgscan_kswapd_dma,pgscan_kswapd_dma32,pgscan_kswapd_normal,pgscan_kswapd_movable,pgscan_direct_dma,pgscan_direct_dma32,pgscan_direct_normal,pgscan_direct_movable,pgscan_direct_throttle,zone_reclaim_failed,pginodesteal,slabs_scanned,kswapd_inodesteal,kswapd_low_wmark_hit_quickly,kswapd_high_wmark_hit_quickly,pageoutrun,allocstall,pgrotated,drop_pagecache,drop_slab,numa_pte_updates,numa_huge_pte_updates,numa_hint_faults,numa_hint_faults_local,numa_pages_migrated,pgmigrate_success,pgmigrate_fail,compact_migrate_scanned,compact_free_scanned,compact_isolated,compact_stall,compact_fail,compact_success,htlb_buddy_alloc_success,htlb_buddy_alloc_fail,unevictable_pgs_culled,unevictable_pgs_scanned,unevictable_pgs_rescued,unevictable_pgs_mlocked,unevictable_pgs_munlocked,unevictable_pgs_cleared,unevictable_pgs_stranded,thp_fault_alloc,thp_fault_fallback,thp_collapse_alloc,thp_collapse_alloc_failed,thp_split,thp_zero_page_alloc,thp_zero_page_alloc_failed,balloon_inflate,balloon_deflate,balloon_migrate,swap_ra,swap_ra_hit".split(',')
eclipse_procstat_col_names = "timestamp,component_id,job_id,app_id,cores_up,cpu_enabled,user,nice,sys,idle,iowait,irq,softirq,steal,guest,guest_nice,hwintr_count,context_switches,processes,procs_running,procs_blocked,softirq_count,per_core_cpu_enabled,per_core_user,per_core_nice,per_core_sys,per_core_idle,per_core_iowait,per_core_irq,per_core_softirqd,per_core_steal,per_core_guest,per_core_guest_nice".split(",")


def convert_str_time_to_unix(str_time):
    
    curr_format = '%Y-%m-%d %H:%M:%S.%f'
    
    datetime_object = datetime.strptime(str_time, curr_format).replace(microsecond=0)
    
    return int(time.mktime(datetime_object.timetuple()))

def add_job_ids(df, job_ids):
    """
        The example sampler CSV's have only one job_id. This function synthetically adds job_ids to the same dataframe
    """
    temp_list = []
    
    for job_id in job_ids:
        df['job_id'] = job_id
        temp_list.append(df.copy(deep=True))
        
    return pd.concat(temp_list)

def transform_dsos_data(meminfo_df, vmstat_df, procstat_df, silent=True):
        
    if not (set(meminfo_df.job_id.unique()) == set(vmstat_df.job_id.unique()) == set(procstat_df.job_id.unique())):
        print(f"WARNING: Provided samplers do not contain the same unique job_ids. The code will try to select the minimal subset of job_ids")
        
    common_job_ids = list((set(meminfo_df.job_id.unique()) & set(vmstat_df.job_id.unique()) & set(procstat_df.job_id.unique())))
    
    training_data_list = []
    for job_id in common_job_ids:        
        single_job_data = transform_dsos_job_data((meminfo_df[meminfo_df['job_id'] == job_id]), (vmstat_df[vmstat_df['job_id'] == job_id]), (procstat_df[procstat_df['job_id'] == job_id]), silent)
        training_data_list.append(single_job_data)            
        
    return pd.concat(training_data_list)


def process_raw_metrics(data, silent=True):
    """Process data based on YAML"""      
    
    if not silent:
        print(f"Processing metrics based on the YAML data")
        
    with open('eclipse_metric_info.yaml', 'r') as f:
        metric_info = yaml.load(f)    
                
    new_data = {}
    for col in data.columns:
        if col not in metric_info:   
            if not silent:
                print("{} not in YAML".format(col))
        elif metric_info[col] == 'cumulative':
            new_data[col] = np.diff(data[col].interpolate()) ## maybe NAN problem when the metric is 0
            if any(new_data[col] < 0):
                if not silent:
                    print("Column {} decreased".format(col))
        elif metric_info[col] in ['important', 'noncumulative']:
            new_data[col] = data[col].interpolate()[1:]
        elif metric_info[col] == 'unknown':
            new_data[col] = data[col].interpolate()[1:]
            if all(np.diff(data[col].interpolate()) >= 0):
                if not silent:
                    print("{} did not decrease".format(col))
        elif metric_info[col] in ['limit', 'unimportant']:
            pass
        else:
            raise IOError("Condition doesn't exist for {}".format(
                metric_info[col]))
    return pd.DataFrame(new_data, index=data.index[1:])
    

def transform_dsos_job_data(meminfo_df, vmstat_df, procstat_df, silent=True):
    
    assert len(meminfo_df['job_id'].unique()) == 1, "All the samplers must contain only one job_id. You can input multiple job_ids using transform_dsos_data"
    assert len(vmstat_df['job_id'].unique()) == 1, "All the samplers must contain only one job_id. You can input multiple job_ids using transform_dsos_data"
    assert len(procstat_df['job_id'].unique()) == 1, "All the samplers must contain only one job_id. You can input multiple job_ids using transform_dsos_data"
    
    curr_job_id = meminfo_df['job_id'].unique()[0]
        
    meminfo_df.drop(columns=junk_cols,inplace=True)
    vmstat_df.drop(columns=junk_cols,inplace=True)    
    procstat_df.drop(columns=junk_cols,inplace=True)
    
    if isinstance(meminfo_df['timestamp'].values[0], str):
        meminfo_df['unix_timestamp'] = meminfo_df['timestamp'].apply(lambda x: convert_str_time_to_unix(x))    
        vmstat_df['unix_timestamp'] = vmstat_df['timestamp'].apply(lambda x: convert_str_time_to_unix(x))    
        procstat_df['unix_timestamp'] = procstat_df['timestamp'].apply(lambda x: convert_str_time_to_unix(x))   
    else:
        meminfo_df['unix_timestamp'] = meminfo_df['timestamp'].astype(int)
        vmstat_df['unix_timestamp'] = vmstat_df['timestamp'].astype(int)
        procstat_df['unix_timestamp'] = procstat_df['timestamp'].astype(int)
        
    sampler_col_names = [curr_col + '::{}'.format("meminfo")  if curr_col not in excluded_cols else curr_col for curr_col in meminfo_df.columns ]
    meminfo_df.columns = sampler_col_names

    sampler_col_names = [curr_col + '::{}'.format("vmstat")  if curr_col not in excluded_cols else curr_col for curr_col in vmstat_df.columns ]
    vmstat_df.columns = sampler_col_names

    sampler_col_names = [curr_col + '::{}'.format("procstat")  if curr_col not in excluded_cols else curr_col for curr_col in procstat_df.columns ]
    procstat_df.columns = sampler_col_names
    
    non_per_core_cols = [curr_col for curr_col in procstat_df.columns if not ('per_core' in curr_col) and not (curr_col in excluded_cols)]
    common_comp_ids = list(set(meminfo_df.component_id.values) & set(procstat_df.component_id.values) & set(vmstat_df.component_id.values))
    
    cleaned_node_data = []

    for comp_id in common_comp_ids:
                
        node_meminfo_df = meminfo_df[meminfo_df['component_id'] == comp_id]
        node_vmstat_df = vmstat_df[vmstat_df['component_id'] == comp_id]
        node_procstat_df = procstat_df[procstat_df['component_id'] == comp_id]
        
        common_time = list(set(node_meminfo_df.unix_timestamp.values) & set(node_procstat_df.unix_timestamp.values) & set(node_vmstat_df.unix_timestamp.values))
        

        node_meminfo_df = node_meminfo_df[node_meminfo_df['unix_timestamp'].isin(common_time)].drop(columns=common_cols)
        node_meminfo_df.set_index('unix_timestamp', inplace=True)

        node_vmstat_df = node_vmstat_df[node_vmstat_df['unix_timestamp'].isin(common_time)].drop(columns=common_cols)
        node_vmstat_df.set_index('unix_timestamp', inplace=True)

        node_procstat_df = node_procstat_df[node_procstat_df['unix_timestamp'].isin(common_time)].drop(columns=common_cols)
        node_procstat_df.set_index('unix_timestamp', inplace=True)
        node_procstat_df = node_procstat_df[non_per_core_cols]

        node_data_df = pd.concat([node_meminfo_df, node_vmstat_df, node_procstat_df],axis=1)                
        
        node_data_df = process_raw_metrics(node_data_df)
                
        node_data_df.index.name = "timestamp"
        node_data_df.reset_index(inplace=True)    
        node_data_df.insert(1, 'job_id', curr_job_id)                
        node_data_df.insert(2, 'component_id', comp_id)
                
        cleaned_node_data.append(node_data_df)
        
        if not silent:
            print(f"Component ID: {comp_id}")
            print(f"Num. unique timestamps for each sampler: {len(node_meminfo_df.unix_timestamp.unique())}, {len(node_vmstat_df.unix_timestamp.unique())}, {len(node_procstat_df.unix_timestamp.unique())}")
            print(f"Common time length: {len(common_time)}")        
    
    return pd.concat(cleaned_node_data)