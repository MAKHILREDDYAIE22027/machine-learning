def calculate_range(number_list):
   
    if len(number_list) < 3:
        raise ValueError("Range determination not possible. The list should have at least 3 elements.")

    list_range = max(number_list) - min(number_list)
    
    return list_range

input_list = [5, 3, 8, 1,0, 4]

try:
    result_range = calculate_range(input_list)
    print(f"The range of the list is: {result_range}")
except ValueError as e:
    print(e)
