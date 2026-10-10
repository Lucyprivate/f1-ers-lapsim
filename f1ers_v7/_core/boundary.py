"""Complete-lap versus physical-stop time labels."""

def label_elapsed_time(result):
    """Do not call a physically stopped partial trajectory's elapsed time a lap."""
    complete=bool(result['summary']['completed_lap'])
    result['summary']['elapsed_time_s']=float(result['terminal_state']['elapsed_time_s'])
    result['summary']['formal_lap_time_s']=float(result['terminal_state']['elapsed_time_s']) if complete else None
    result['summary']['time_interpretation']='complete_lap' if complete else 'partial_elapsed_to_stop'
    if not complete:
        result['summary']['lap_time_s']=None
    return result
