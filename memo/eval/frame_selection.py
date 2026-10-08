"""Frame-budget policies, independent of model packages."""
def uniform_sample(items, count):
    if count <= 0 or not items: return []
    count=min(count,len(items))
    if count==1: return [items[0]]
    return [items[i*(len(items)-1)//(count-1)] for i in range(count)]

def dual_budget(history, current, history_frames, current_frames):
    return uniform_sample(history,history_frames)+uniform_sample(current,current_frames)
