def count_with_sum_10(nums):
    count=0
    seen=set()
    for num in nums:
        compliment=10-num
        if compliment in seen:
            count +=1
            seen.add(num)
    return count
nums=[2,7,4,1,3,6]
pairs_count=count_with_sum_10(nums)
print("pairs with sum equal to 10:",pairs_count)