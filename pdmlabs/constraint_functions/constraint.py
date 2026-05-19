import math


def self_tuning_constraint_function(current_pipeline):
    def nested_function(params_configuration_list):
        result = []
        for param_configuration_dict in params_configuration_list:
            if "incremental_slide" in param_configuration_dict.keys():
                #  TODO self_tunning_size + init < num_records_upper_limit
                result.append(param_configuration_dict['incremental_slide'] > param_configuration_dict['postprocessor_window_length'] and param_configuration_dict['postprocessor_window_length'] <= current_pipeline.dataset['max_wait_time'])
            elif "profile_size" in param_configuration_dict.keys():
                result.append(param_configuration_dict['profile_size'] + param_configuration_dict['postprocessor_window_length'] <= current_pipeline.dataset['max_wait_time'])
            else:
                # semi-supervised and unsupervised
                result.append(param_configuration_dict['postprocessor_window_length'] <= current_pipeline.dataset['max_wait_time'])
        return result

    return nested_function


def auto_profile_max_wait_time_constraint(current_pipeline):
    def nested_function(params_configuration_list):
        result = []
        for param_configuration_dict in params_configuration_list:
            if param_configuration_dict['profile_size'] <= current_pipeline.dataset['max_wait_time']:
                result.append(True)
            else:
                result.append(False)

        return result

    return nested_function


def incremental_max_wait_time_constraint(current_pipeline):
    def nested_function(params_configuration_list):
        result = []
        for param_configuration_dict in params_configuration_list:
            if param_configuration_dict['initial_incremental_window_length'] <= current_pipeline.dataset['max_wait_time']:
                result.append(True)
            else:
                result.append(False)
                
        return result

    return nested_function


def unsupervised_max_wait_time_constraint(current_pipeline):
    def nested_function(params_configuration_list):
        result = []
        for param_configuration_dict in params_configuration_list:
            tempResult=True
            if "method_window" in param_configuration_dict.keys() and "method_slide" in param_configuration_dict.keys():
                tempResult= tempResult and (param_configuration_dict['method_window'] <= current_pipeline.dataset['max_wait_time'])

            if "method_context_length" in param_configuration_dict.keys() and "method_slide" in param_configuration_dict.keys():
                tempResult= tempResult and (param_configuration_dict['method_context_length'] <= current_pipeline.dataset['max_wait_time'])

            if "method_window" in param_configuration_dict.keys() and "method_sub_sequence_length" in param_configuration_dict.keys():
                tempResult = tempResult and (param_configuration_dict['method_sub_sequence_length'] < param_configuration_dict['method_window'])

            if "method_init_length" in param_configuration_dict.keys() and "method_sub_sequence_length" in param_configuration_dict.keys():
                tempResult = tempResult and (param_configuration_dict['method_sub_sequence_length'] < current_pipeline.dataset['method_init_length'])

            result.append(tempResult)
            
        return result
    
    return nested_function


def NP_semi_supervised():
    def nested_function(params_configuration_list):
        result = []
        for param_configuration_dict in params_configuration_list:
            if "incremental_slide" in param_configuration_dict.keys():
                result.append(param_configuration_dict['incremental_slide'] > param_configuration_dict[
                    'method_sub_sequence_length'])
        else:
            result.append(True)
        return result

    return nested_function


def combine_constraint_functions(*functions):
    def nested_function(params_configuration_list):
        initial_results = []
        for function in functions:
            initial_results.append(function(params_configuration_list))
        
        result = []
        for column in zip(*initial_results):
            if False in column:
                result.append(False)
            else:
                result.append(True)
            
        return result
    
    return nested_function


def sand_parameters_constraint_function(current_pipeline=None):
    """

    Returns
    ------- Function Reference which compute the valid parametrization for SAND.

            The Nested Function return a Boolean list which refer to params_configuration_list,
            where True indicates that the parameter combination is valid, and False indicates that the
            parameters are invalid. The calculated results list has length equal to the  params_configuration_list
            length.

    """
    def nested_function_sand(params_configuration_list):
        result = []
        for param_configuration_dict in params_configuration_list:
            if param_configuration_dict["method_init_length"] < param_configuration_dict["method_batch_size"]:
                result.append(False)
                continue

            if param_configuration_dict["method_batch_size"] < param_configuration_dict["method_subsequence_length_multiplier"] * param_configuration_dict["method_pattern_length"]:
                result.append(False)
                continue

            current_subsequence_length = current_overlapping_rate = param_configuration_dict['method_subsequence_length_multiplier'] * param_configuration_dict['method_pattern_length']
            if current_overlapping_rate > param_configuration_dict['method_batch_size']:
                result.append(False)
                continue
            
            if current_pipeline is not None: # NOTE: added for backwards compatibility
                if current_overlapping_rate > current_pipeline.dataset['min_target_scenario_len'] // 2:
                    result.append(False)
                    continue
            
            # NOTE: also subsequence_length must be greater than pattern_length which in our implementation is by default true because we use the subsequence_length_multiplier
            if param_configuration_dict["method_init_length"] > current_pipeline.dataset['max_wait_time']:
                result.append(False)
                continue

            # if param_configuration_dict["method_init_length"] >= current_pipeline.dataset['min_target_scenario_len'] - current_subsequence_length:
            #     result.append(False)
            #     continue

            # if float(param_configuration_dict["method_subsequence_length"]/param_configuration_dict["method_pattern_length"])!=float(param_configuration_dict["method_subsequence_length"]//param_configuration_dict["method_pattern_length"]):
            #     result.append(False)
            #     continue
            result.append(True)

        return result
    
    return nested_function_sand


def incremental_constraint_function(params_configuration_list):
    result = []

    for param_configuration_dict in params_configuration_list:
        result.append(
            param_configuration_dict['incremental_window_length'] >= param_configuration_dict['initial_incremental_window_length']
            and param_configuration_dict['incremental_slide'] <= param_configuration_dict['initial_incremental_window_length']
        )

    return result


def unsupervised_distance_based(params_configuration_list):
    result = []

    for param_configuration_dict in params_configuration_list:
        result.append(param_configuration_dict['method_window'] > param_configuration_dict['method_k'])

    return result
