import pandas as pd


class ELSplitter:

    def __init__(self, print_stats=False):
        self.print_stats = print_stats

    def _remove_outliers(self, el, start_date, end_date):
        case_durations = el.groupby('case:concept:name')['time:timestamp'].agg(['min', 'max'])
        case_durations['duration'] = case_durations['max'] - case_durations['min']
        duration_95th_percentile = case_durations['duration'].quantile(0.95)
        valid_cases = case_durations[case_durations['duration'] <= duration_95th_percentile].index
        cases_original = el['case:concept:name'].nunique()
        el = el[el['case:concept:name'].isin(valid_cases)]
        if self.print_stats:
            print(f"Cases removed after removing top 5%: {cases_original - el['case:concept:name'].nunique()}")

        case_times = el.groupby('case:concept:name')['time:timestamp'].agg(['min', 'max']).reset_index()
        if el['time:timestamp'].dt.tz is not None:
            if start_date is not None:
                start_date = pd.to_datetime(start_date).tz_localize(el['time:timestamp'].dt.tz)
            if end_date is not None:
                end_date = pd.to_datetime(end_date).tz_localize(el['time:timestamp'].dt.tz)
        else:
            if start_date is not None:
                start_date = pd.to_datetime(start_date)
            if end_date is not None:
                end_date = pd.to_datetime(end_date)

        data_min = el['time:timestamp'].min()
        data_max = el['time:timestamp'].max()

        if start_date is not None:
            if start_date < data_min:
                print(f"Warning: start_date {start_date} is earlier than the earliest event log timestamp {data_min}.")
                start_date = data_min
        if end_date is not None:
            if end_date > data_max:
                print(f"Warning: end_date {end_date} is later than the latest event log timestamp {data_max}.")
                end_date = data_max

        case_mask = pd.Series([True] * len(case_times), index=case_times.index)
        if start_date is not None:
            case_mask &= case_times['min'] >= start_date
        if end_date is not None:
            case_mask &= case_times['max'] <= end_date
        valid_cases = case_times[case_mask]['case:concept:name']
        filtered_df = el[el['case:concept:name'].isin(valid_cases)]

        if self.print_stats:
            print(
                f"Cases removed after removing chronological outliers: {el['case:concept:name'].nunique() - filtered_df['case:concept:name'].nunique()}")
        return filtered_df

    def _debias_test_end(self, el):
        case_durations = el.groupby('case:concept:name')['time:timestamp'].agg(['min', 'max'])
        case_durations['duration'] = case_durations['max'] - case_durations['min']
        longest_duration = case_durations['duration'].max()

        end_date = el['time:timestamp'].max()

        test_end_date = end_date - longest_duration
        filtered_el = el[el['time:timestamp'] <= test_end_date]

        if self.print_stats:
            print(f"N.o. events removed from test set end: {len(el) - len(filtered_el)}")

        return filtered_el

    def _find_case_fraction_complete(self, original_el, test_df):
        original_counts = original_el.groupby('case:concept:name').size().rename('total_events')
        original_el['event_index'] = original_el.groupby('case:concept:name').cumcount() + 1  # 1-based index
        original_el = original_el.merge(original_counts, on='case:concept:name')
        original_el['fraction_complete'] = original_el['event_index'] / original_el['total_events']
        test_df = test_df.merge(
            original_el[['fraction_complete']],
            how='left',
            left_index=True,
            right_index=True
        )

        return test_df

    def _find_split_timestamp(self, el, test_length=0.2):
        case_start_times = el.groupby('case:concept:name')['time:timestamp'].min().reset_index()
        case_start_times = case_start_times.sort_values('time:timestamp')
        split_index = int(1 - test_length * len(case_start_times))
        split_timestamp = case_start_times.iloc[split_index]['time:timestamp']
        return split_timestamp, split_index

    def _create_train(self, el, split_timestamp):
        case_end_times = el.groupby('case:concept:name')['time:timestamp'].max().reset_index()
        valid_cases = case_end_times[case_end_times['time:timestamp'] < split_timestamp]['case:concept:name']
        train = el[el['case:concept:name'].isin(valid_cases)]
        if self.print_stats:
            num_train_cases = el[el['time:timestamp'] <= split_timestamp]['case:concept:name'].nunique()
            removed_cases = num_train_cases - train['case:concept:name'].nunique()
            print(f"% Cases removed from train set: {(removed_cases / num_train_cases) * 100}%")
        return train

    def _create_test(self, el, split_timestamp):
        test = el[el['time:timestamp'] >= split_timestamp]
        return test

    def debias_and_split(self, el, start_date, end_date):
        if self.print_stats:
            print(f"Number of cases before pre-processing: {el['case:concept:name'].nunique()}")
        original_el = el.copy()
        el = self._remove_outliers(el, start_date, end_date)
        el = self._debias_test_end(el)
        split_timestamp, split_index = self._find_split_timestamp(el)
        train = self._create_train(el, split_timestamp)
        test = self._create_test(el, split_timestamp)
        test_with_fraction_complete = self._find_case_fraction_complete(original_el, test)

        if self.print_stats:
            print(
                f"Number of cases after pre-processing: {train['case:concept:name'].nunique() + test['case:concept:name'].nunique()}")
            print(f"Number of cases train: {train['case:concept:name'].nunique()}")
            print(f"Number of cases test: {test['case:concept:name'].nunique()}")
        return train, test_with_fraction_complete
