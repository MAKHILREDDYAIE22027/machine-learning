def count_highest_occurrence(input_string):
    
    char_count = {}

    clean_string = ''.join(char.lower() for char in input_string if char.isalpha())

    
    for char in clean_string:
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1

    max_char = max(char_count, key=char_count.get)
    max_count = char_count[max_char]

    return max_char, max_count

input_str = "hippopotamus"

max_char, max_count = count_highest_occurrence(input_str)

print(f"The highest occurring character is '{max_char}' with a count of {max_count}.")
