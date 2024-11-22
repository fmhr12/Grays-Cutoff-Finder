import pandas as pd
import numpy as np
import math

def find_best_cutoffs(df, predictor, time_col, event_col, competing_event_code=2):
    """
    Finds the best cutoff points for a continuous predictor variable by maximizing standardized Gray's test statistic.

    Parameters:
    -----------
    df : pandas DataFrame
        The dataset containing the predictor variable, survival times, and event indicators.
    predictor : str
        The name of the continuous prognostic factor in df.
    time_col : str
        The name of the column in df containing survival times.
    event_col : str
        The name of the column in df containing event indicators.
        Event indicators should be coded as:
            - 0: Censored
            - 1: Event of interest
            - competing_event_code: Competing event
    competing_event_code : int, optional
        The code representing competing events in event_col (default is 2).

    Returns:
    --------
    results_df : pandas DataFrame
        A DataFrame containing cutoff values and corresponding standardized Gray's test statistics,
        sorted by the absolute value of the test statistic in descending order.
        Columns include:
            - 'Cutoff': The cutoff value for the predictor variable.
            - 'U_tilde': The Gray's test statistic numerator.
            - 'Z': The standardized Gray's test statistic.
            - 'P_value': The approximated p-value for the test statistic.
    """

    # Ensure the necessary columns are present in the DataFrame
    required_columns = [predictor, time_col, event_col]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the DataFrame.")

    # Copy the DataFrame to avoid modifying the original data
    df = df.copy()

    # Predictor variable (continuous prognostic factor)
    # Optionally, scale the predictor if necessary
    # df[predictor] = df[predictor] / scaling_factor  # Uncomment and set scaling_factor if needed

    # Survival times
    df['T'] = df[time_col]

    # Event indicators
    df['delta'] = df[event_col]

    # Get all unique, sorted values of the prognostic factor as potential cutoffs
    cutoff_values = np.sort(df[predictor].unique())

    # Initialize a list to store results
    results = []

    # Number of subjects
    n = len(df)

    # Identify times when an event of interest has occurred
    event_times = np.sort(df.loc[df['delta'] == 1, 'T'].unique())
    k = len(event_times)

    # Loop over all possible cutoff values
    for cutoff in cutoff_values:
        # Binarize the prognostic factor at the current cutoff to create groups g = 0 and g = 1
        df['g'] = np.where(df[predictor] < cutoff, 0, 1)

        # Skip if all subjects are in one group
        if df['g'].nunique() < 2:
            continue

        # Initialize dictionaries to store counts and estimates
        d_gi = {0: [], 1: []}  # Number of events of interest in group g at time t_i
        r_gi = {0: [], 1: []}  # Number at risk in group g at time t_i
        c_gi = {0: [], 1: []}  # Number of competing events in group g at time t_i
        r_i_list = []          # Total number at risk at time t_i
        d_i_list = []          # Total number of events of interest at time t_i
        r_tilde_gi = {0: [], 1: []}  # Adjusted risk sets for group g at time t_i
        r_tilde_i_list = []          # Total adjusted risk set at time t_i
        w_gi = {0: [], 1: []}        # Weights for group g at time t_i
        S_hat_g = {0: [1.0], 1: [1.0]}  # Estimated survival function for group g
        F_hat_g = {0: [0.0], 1: [0.0]}  # Estimated CIF for group g

        # Compute counts at each event time t_i
        for i, t_i in enumerate(event_times):
            r_i = 0  # Total number at risk at time t_i
            d_i = 0  # Total number of events of interest at time t_i
            for g in [0, 1]:
                # Number at risk in group g at time t_i
                at_risk = df[(df['g'] == g) & (df['T'] >= t_i)]
                r_gi[g].append(len(at_risk))

                # Number of events of interest in group g at time t_i
                events = df[(df['g'] == g) & (df['delta'] == 1) & (df['T'] == t_i)]
                d_gi[g].append(len(events))

                # Number of competing events in group g at time t_i
                comps = df[(df['g'] == g) & (df['delta'] == competing_event_code) & (df['T'] == t_i)]
                c_gi[g].append(len(comps))

                r_i += len(at_risk)
                d_i += len(events)
            r_i_list.append(r_i)
            d_i_list.append(d_i)

        # Compute the estimated CIF and survival function for each group
        for i in range(k):
            for g in [0, 1]:
                # Compute weight w_{gi}
                F_prev = F_hat_g[g][-1]
                S_prev = S_hat_g[g][-1]
                if S_prev > 0:
                    w_gi_gi = (1 - F_prev) / S_prev
                else:
                    w_gi_gi = 0.0
                w_gi[g].append(w_gi_gi)

                # Update survival function S_hat_g(t_i)
                rgi = r_gi[g][i]
                if rgi > 0:
                    hazard = (d_gi[g][i] + c_gi[g][i]) / rgi
                    S_next = S_prev * (1 - hazard)
                else:
                    S_next = S_prev
                S_hat_g[g].append(S_next)

                # Update CIF F_hat_g(t_i)
                if rgi > 0:
                    inc = S_prev * (d_gi[g][i] / rgi)
                else:
                    inc = 0.0
                F_next = F_prev + inc
                F_hat_g[g].append(F_next)

        # Compute adjusted risk sets \tilde{r}_{gi}
        for i in range(k):
            r_tilde_i = 0.0
            for g in [0, 1]:
                r_tilde_gi_gi = w_gi[g][i] * r_gi[g][i]
                r_tilde_gi[g].append(r_tilde_gi_gi)
                r_tilde_i += r_tilde_gi_gi
            r_tilde_i_list.append(r_tilde_i)

        # Compute Gray's statistic \tilde{U} and variance \tilde{V}
        U_tilde = 0.0
        V_tilde = 0.0
        for i in range(k):
            d_1i = d_gi[1][i]
            d_i = d_gi[0][i] + d_gi[1][i]
            r_tilde_1i = r_tilde_gi[1][i]
            r_tilde_i = r_tilde_i_list[i]

            # Compute numerator term
            if r_tilde_i > 0:
                term = d_1i - d_i * (r_tilde_1i / r_tilde_i)
            else:
                term = 0.0
            U_tilde += term

            # Compute variance component
            r_i = r_gi[0][i] + r_gi[1][i]
            r_tilde_0i = r_tilde_gi[0][i]
            numerator = r_tilde_0i * r_tilde_1i * d_i * (r_i - d_i)
            denominator = (r_tilde_i ** 2) * (r_i - 1)
            if denominator > 0:
                V_i = numerator / denominator
            else:
                V_i = 0.0
            V_tilde += V_i

        # Compute the standardized Gray's statistic Z
        if V_tilde > 0:
            Z = U_tilde / np.sqrt(V_tilde)
        else:
            Z = 0.0

        # Compute p-value using an approximation method
        def p_value_approximation(q, terms=10):
            """
            Approximates the p-value based on the standardized statistic q.
            For q > 1, uses a large-q approximation.
            For q <= 1, uses a series expansion approximation.

            Parameters:
            -----------
            q : float
                The standardized test statistic.
            terms : int
                The number of terms to include in the series expansion (default is 10).

            Returns:
            --------
            p_value : float
                The approximated p-value.
            """
            if q > 1:
                return 2 * math.exp(-2 * q ** 2)
            else:
                # For q <= 1, use the series expansion approximation
                p_value = 0
                for j in range(1, terms + 1):
                    term = (-1) ** (j + 1) * math.exp(-2 * (j ** 2) * (q ** 2))
                    p_value += term
                return max(2 * p_value, 0.30)  # Ensure p-value is at least 0.30 for small q

        p_value = p_value_approximation(Z)

        # Store the results
        results.append({
            'Cutoff': cutoff,
            'U_tilde': U_tilde,
            'Z': Z,
            'P_value': p_value
        })

    # Create a DataFrame with the results and sort by absolute Z in descending order
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df['abs_Z'] = results_df['Z'].abs()
        results_df.sort_values(by='abs_Z', ascending=False, inplace=True)
        results_df.reset_index(drop=True, inplace=True)
    else:
        print("No valid cutoffs found.")
        return pd.DataFrame()

    return results_df[['Cutoff', 'U_tilde', 'Z', 'P_value']]

# Example usage
if __name__ == "__main__":
    # Load the data
    # Replace 'your_file_path.xlsx' and 'your_sheet_name' with actual file path and sheet name
    file_path = 'your_file_path.xlsx'
    sheet_name = 'your_sheet_name'
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Define parameters
    predictor = 'your_predictor_column'
    time_col = 'your_time_column'
    event_col = 'your_event_indicator_column'
    competing_event_code = 2  # Change if different in your dataset

    # Call the function
    results_df = find_best_cutoffs(df, predictor, time_col, event_col, competing_event_code)

    # Output the table of best cutoff points
    print("Table of Cutoff Points Sorted by Gray's Statistic (in descending order):")
    print(results_df)
