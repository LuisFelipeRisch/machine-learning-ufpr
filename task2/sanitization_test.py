import unittest
from sanitization import sanitize_data
from random import randint
import math

LAST_MONTH_DAY = 31

class TestSanitization(unittest.TestCase): 
    def test_sanitized_data_should_not_contain_days(self):
        days = range(1, LAST_MONTH_DAY + 1)
        for day in days:
            sanitized_data_df = sanitize_data(desired_day=day)

            if day == 1: 
                self.assertTrue(sanitized_data_df.empty) 
                continue

            should_not_contain_days = range(day, LAST_MONTH_DAY + 1)
            should_not_contain_days_is_in_sanitized_data = False
            for should_not_contain_day in should_not_contain_days: 
                if should_not_contain_day in sanitized_data_df['DAY'].values: 
                    should_not_contain_days_is_in_sanitized_data = True 
                    break

            self.assertFalse(should_not_contain_days_is_in_sanitized_data)
    
    def test_tp_est_column_should_not_contain_zero_value_neither_nan(self):
        day = randint(2, LAST_MONTH_DAY)
        sanitized_data_df = sanitize_data(desired_day=day)
        self.assertFalse(sanitized_data_df['TP_EST'].isnull().values.any())
        
        for value in sanitized_data_df['TP_EST'].values:
            self.assertFalse(math.isclose(value, 0.0))
    
    def test_all_columns_except_tp_est_and_date_columns_should_not_contain_nan(self):
        day = randint(2, LAST_MONTH_DAY)
        sanitized_data_df = sanitize_data(desired_day=day)
        sanitized_data_df = sanitized_data_df.drop(["TP_EST", "DATA"], axis = 1)
        self.assertFalse(sanitized_data_df.isnull().values.any())

if __name__ == '__main__': 
    unittest.main()