import os
import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from colorsys import rgb_to_hls
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
from scipy.integrate import simpson
from scipy.optimize import curve_fit
from IPython.display import Markdown
from sklearn.metrics import r2_score
matplotlib.rc('font', family='Times New Roman')

def is_dark_color(hex_color):
    rgb = mcolors.hex2color(hex_color)
    h, l, s = rgb_to_hls(*rgb)
    return l < 0.5

def calculate_shift_factors(data):
    data['Frequency (Hz)'] = data['Frequency (Hz)'].astype(float)
    data['Storage Modulus (MPa)'] = data['Storage Modulus (MPa)'].astype(float)

    # Extract unique temperatures
    temperatures = data['Temperature (C)'].unique()
    
    # Reference temperature
    T_ref = data['Temperature (C)'].mean()  # Using the mean temperature as the reference
    closest_temp = data['Temperature (C)'].iloc[(data['Temperature (C)'] - T_ref).abs().argsort()[:1]].values[0]

    # Group data by temperature
    grouped = data.groupby('Temperature (C)')

    # Calculate shift factors (a_T) using the frequency at the reference temperature
    reference_frequency = grouped.get_group(closest_temp)['Frequency (Hz)'].mean()
    shift_factors = {}
    for temp, group in grouped:
        shift_factors[temp] = reference_frequency / group['Frequency (Hz)'].mean()

    # Logarithm of shift factors
    log_aT = np.log10(list(shift_factors.values()))
    T_diff = np.array(list(shift_factors.keys())) - closest_temp

    # Define the WLF equation
    def wlf_equation(T_diff, C1, C2):
        return -C1 * T_diff / (C2 + T_diff)

    # Initial guess for C1 and C2
    initial_guess = [17.44, 51.6]

    # Fit the WLF equation
    params, covariance = curve_fit(wlf_equation, T_diff, log_aT, p0=initial_guess)
    C1, C2 = params

    # Reference temperature (use the closest actual temperature to the mean as reference)
    T_ref = data['Temperature (C)'].mean()
    closest_temp = data['Temperature (C)'].iloc[(data['Temperature (C)'] - T_ref).abs().argsort()[:1]].values[0]

    # Constants C1 and C2 from the fit
    # C1 = -0.032
    # C2 = 321.16

    # Calculate shift factors
    def calculate(T, T_ref, C1, C2):
        return 10 ** (-C1 * (T - T_ref) / (C2 + (T - T_ref)))

    data['Shift Factor'] = data['Temperature (C)'].apply(lambda T: calculate(T, closest_temp, C1, C2))

    # Display the shift factors
    print(data[['Temperature (C)', 'Shift Factor']])

    return shift_factors

def process_csv(file_path, shift_factors, use_equation):
    # Convert shift_factors from string to dictionary if not using equation
    if not use_equation:
        shift_factors = eval(shift_factors)
    else:
        data = pd.read_csv(file_path)
        shift_factors = calculate_shift_factors(data)

    # Load the data and convert to floating point
    data = pd.read_csv(file_path)
    data['Frequency (Hz)'] = data['Frequency (Hz)'].astype(float)
    data['Storage Modulus (MPa)'] = data['Storage Modulus (MPa)'].astype(float)

    # Extract unique temperatures
    temperatures = data['Temperature (C)'].unique()

    # Getting darker colors from matplotlib
    all_colors = list(mcolors.CSS4_COLORS.values())
    darker_colors = [color for color in all_colors if is_dark_color(color)]

    # Setting up the plot
    plt.figure(figsize=(12, 8))

    # Plot each temperature's data with a darker color
    for i, temp in enumerate(temperatures):
        subset = data[data['Temperature (C)'] == temp]
        color = darker_colors[i % len(darker_colors)]  # Cycle through the darker color list
        plt.semilogx(subset['Frequency (Hz)'], subset['Storage Modulus (MPa)'], color=color, label=f'{temp} °C')

    label_font = {'fontname': 'Times New Roman', 'size': 26}
    plt.xlabel('Frequency (Hz)', **label_font)
    plt.ylabel('Storage Modulus (MPa)', **label_font)
    plt.title('Storage Modulus vs. Frequency at Different Temperatures', **label_font)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(1e-1, 10)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Define the path to save the figure
    output_dir = 'uploads'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'plot1.png')
    plt.savefig(output_path)
    plt.close()

    ######################
    ######################

    ### Surface plot  ###
    # Create grid for interpolation
    Y = data['Frequency (Hz)']
    X = data['Temperature (C)']
    Z = data['Storage Modulus (MPa)']
    X_grid, Y_grid = np.meshgrid(np.linspace(X.max(), X.min(), 100), np.linspace(Y.min(), Y.max(), 100))
    Z_grid = griddata((X, Y), Z, (X_grid, Y_grid), method='cubic')

    # Plotting
    fig = plt.figure(figsize=(28, 12))
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='rainbow', edgecolor='none')

    ax.set_ylabel('Frequency (Hz)', fontsize=24)
    ax.set_xlabel('Temperature (C)', fontsize=24)
    ax.set_zlabel('Storage Modulus (MPa)', fontsize=24)
    ax.set_title('3D Surface Plot: Storage Modulus vs Frequency and Temperature', fontsize=30)
    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=20)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'plot2.png')
    plt.savefig(output_path)
    plt.close()

    ######################
    ######################

    ### Master Curve Plot ###
    combined_log_freq = []
    combined_storage_modulus = []

    for temp in data['Temperature (C)'].unique():
        temp_data = data[data['Temperature (C)'] == temp]
        if temp in shift_factors:
            shift_factor = shift_factors[temp]
            shifted_log_freq = np.log10(temp_data['Frequency (Hz)'] * shift_factor)
            combined_log_freq.extend(shifted_log_freq)
            combined_storage_modulus.extend(temp_data['Storage Modulus (MPa)'])
        else:
            print(f"Warning: Temperature {temp} not found in shift_factors dictionary.")
            continue

    combined_log_freq = np.array(combined_log_freq)
    combined_storage_modulus = np.array(combined_storage_modulus)

    def storage_modulus_model(log_omega, a, b, c, d):
        return a * np.tanh(b * (log_omega + c)) + d

    lower_bounds = [1e-6, -100, -100, 1e-6]
    upper_bounds = [100, 100, 100, 100]

    params, _ = curve_fit(
        storage_modulus_model,
        combined_log_freq,
        combined_storage_modulus,
        bounds=(lower_bounds, upper_bounds),
        maxfev=10000
    )

    a, b, c, d = params

    plt.figure(figsize=(14, 10))

    for i, temp in enumerate(sorted(data['Temperature (C)'].unique())):
        if temp in shift_factors:
            subset = data[data['Temperature (C)'] == temp]
            shift_factor = shift_factors[temp]
            shifted_log_freq = np.log10(subset['Frequency (Hz)'] * shift_factor)
            storage_modulus = subset['Storage Modulus (MPa)']
            color = darker_colors[i % len(darker_colors)]
            plt.scatter(shifted_log_freq, storage_modulus, color=color, label=f'{temp} °C')

    log_freq_range = np.linspace(min(combined_log_freq), max(combined_log_freq), 1000)
    fitted_storage_modulus = storage_modulus_model(log_freq_range, a, b, c, d)
    plt.plot(log_freq_range, fitted_storage_modulus, label='Fitted Model', color='r', linestyle='--', linewidth=3)

    plt.xlabel("Reduced Frequency log(Hz)", fontsize=40)
    plt.ylabel(r"$\it{E}'$ (MPa)", fontsize=40, fontname='Times New Roman')
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(fontsize=20)

    output_path = os.path.join(output_dir, 'plot4.png')
    plt.savefig(output_path)
    plt.close()

    print(f"Estimated parameters:\nA = {a}\nB = {b}\nC = {c}\nD = {d}")

    ######################
    ######################

    A = a
    B = b
    C = c
    D = d

    # Define the function E'(w)
    def E_prime(w):
        return A * np.tanh(B * ((np.log(w)) + C)) + D

    # Generate a range of w values on a logarithmic scale
    w = np.logspace(-20, 10, 500)  # Range from 0.01 to 100

    def tanh_value(w):
        return np.tanh(B * (np.log(w) + C))

    tanh_v = tanh_value(w)

    E_prime_values = E_prime(w)

    # Plotting E'(w)
    plt.figure(figsize=(12, 8))
    plt.plot(w, E_prime_values, label="E'(w)")
    plt.xlabel('Frequency (w) (Hz)')
    plt.ylabel("E'(w) (MPa)")
    plt.title("Plot of E'(w)")
    plt.xscale('log')  # Logarithmic scale for frequency
    plt.grid(True)
    plt.legend()

    output_path = os.path.join(output_dir, 'plot5.png') 
    plt.savefig(output_path)
    plt.close()

    ######################
    ######################

    ### Single E(t) Plot ###

    # Function time_cycle for Etime
    def Etime_time_cycle(time, cycle=500):
        
        # Integration of sigmoidal curve numerically to get E(t)
        # Number of Sample Points per cycle
        N1, N2, N3 = 240, 74, 24

        # Initialize Etime
        Etime = np.zeros_like(time)

        # Define integrand function
        def integrand(t,E_prime_w, w):
            return (2/np.pi)*(E_prime_w/w)*np.sin(w*t)

        # Process each time point
        for i, t in enumerate(time):
            # Conditions for different ranges of cycles
            w1 = np.linspace((1e-6 / t), (cycle * 0.1 * 2 * np.pi / t), int(cycle * 0.1 * N1 + 1))
            w2 = np.linspace((cycle * 0.1 * 2 * np.pi) / t, (cycle * 0.4 * 2 * np.pi) / t, int(cycle * 0.3 * N2 + 1))
            w3 = np.linspace((cycle * 0.4 * 2 * np.pi) / t, (cycle * 2 * np.pi) / t, int(cycle * 0.6 * N3 + 1))

            # Concatenating all frequencies
            all_w = np.concatenate([w1, w2[1:], w3[1:]])  # Avoid repeating points at boundaries

            # Compute integrand for all frequencies
            y = integrand(t, E_prime(all_w), all_w)

            # Integrate using the trapezoidal method
            Etime[i] = np.trapezoid(y, all_w)

        return Etime

    # Generate a time array (logarithmic scale)
    time = np.logspace(-10, 10, 500)  # 500 points from 1e-20 to 1e20

    # Compute Etime
    Etime = Etime_time_cycle(time)

    # Plotting Etime vs time on a linear scale with adjusted y-axis
    plt.figure(figsize=(12, 8))
    plt.plot(time, Etime, label='E(t) vs Time')
    plt.xlabel('Time (s)', fontsize=24)
    plt.ylabel('Relaxation Modulous (MPa)', fontsize=24)
    plt.title('Relaxation Modulous with Time', fontsize=24)
    plt.xscale('log')  # Keeping the x-axis on a logarithmic scale for better visibility
    plt.xticks(fontsize=18)
    plt.xlim(1e-8,1e10)
    plt.yticks( fontsize=18)

    output_path = os.path.join(output_dir, 'plot6.png')
    plt.savefig(output_path)
    plt.close()

    ######################
    ######################

    ### Plot the elastic modulus vs strain rate at Ref_temperature and Secant Modulus ###

    # Define Secant Modulus and strain rates
    strain_limit = 0.0025 # 0.25% Secant Modulus
    strain_rates = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]

    # Initialize array for elastic modulus
    elastic_modulus = []

    # Calculate the stress at the strain limit for each strain rate
    for rate in strain_rates:
        # Time corresponding to the strain limit for the current strain rate
        time_at_limit = strain_limit / rate

        # Compute E(t) at this time
        Etime_at_limit = Etime_time_cycle(np.array([time_at_limit]))

        # The stress at the strain limit is E(t) * strain
        stress_at_limit = Etime_at_limit[0] * strain_limit

        # Elastic modulus is stress/strain
        modulus = stress_at_limit / strain_limit
        elastic_modulus.append(modulus)

    estimated_shift_factors = {}
    for reference_temp in shift_factors.keys():
        scale_factor = 1 / shift_factors[reference_temp]
        estimated_shift_factors[reference_temp] = {temp: factor * scale_factor for temp, factor in shift_factors.items()}

    # Creating a new figure for the plot
    plt.figure(figsize=(12, 8))

    # Define a colormap
    n_ref_temps = len(estimated_shift_factors.keys())
    colors = cm.rainbow(np.linspace(0, 1, n_ref_temps))

    # Looping over each reference temperature
    for ref_temp, color in zip(sorted(estimated_shift_factors.keys()), colors):
        ref_shift_factors = estimated_shift_factors[ref_temp]

        # Plotting each temperature's data
        for temp in sorted(temperatures):
            if temp in ref_shift_factors:
                # Filter data for the current temperature
                subset = data[data['Temperature (C)'] == temp].copy()
                # Apply the shift factor
                subset['Adjusted Frequency (Hz)'] = subset['Frequency (Hz)'] * ref_shift_factors[temp]
                # Plot
                plt.semilogx(subset['Adjusted Frequency (Hz)'], subset['Storage Modulus (MPa)'], 'o', color=color, label=f'Ref Temp {ref_temp}°C' if temp == min(temperatures) else "")

    # Adding labels, title, grid, and legend
    plt.xlabel('Reduced Frequency (Hz)', fontsize=24)
    plt.ylabel('Storage Modulus (MPa)', fontsize=24)
    plt.title('Master Curve for All Reference Temperatures', fontsize=24)
    plt.grid(True)
    plt.legend()

    output_path = os.path.join(output_dir, 'plot8.png')  # Define the full path
    plt.savefig(output_path)
    plt.close()

    ######################
    ######################

    # Define the model function based on the provided equation
    def storage_modulus_model(log_omega, a, b, c, d):
        return a * np.tanh(b * (log_omega + c)) + d

    # Define bounds for the curve fitting parameters
    lower_bounds = [0, -100, -100, 0]
    upper_bounds = [100, 100, 100, 100]

    # Creating a new figure for the plot
    plt.figure(figsize=(14, 10))

    # Define a colormap
    n_ref_temps = len(estimated_shift_factors.keys())
    colors = cm.rainbow(np.linspace(0, 1, n_ref_temps))

    # List to store DataFrames for each reference temperature
    all_data_frames = []
    # List to store parameters for each reference temperature
    abcd_parameters = []
    # Dictionary to store the Coefficient of Determination (R^2) for each fitting
    r2_scores = {}

    # Looping over each reference temperature for plotting and fitting
    for ref_temp, color in zip(sorted(estimated_shift_factors.keys()), colors):
        ref_shift_factors = estimated_shift_factors[ref_temp]
        combined_log_freq = []
        combined_storage_modulus = []

        # Plotting each temperature's data
        for temp in sorted(temperatures):
            if temp in ref_shift_factors:
                subset = data[data['Temperature (C)'] == temp].copy()
                subset['Adjusted Frequency (Hz)'] = subset['Frequency (Hz)'] * ref_shift_factors[temp]
                combined_log_freq.extend(np.log10(subset['Adjusted Frequency (Hz)']))
                combined_storage_modulus.extend(subset['Storage Modulus (MPa)'])

                plt.semilogx(subset['Adjusted Frequency (Hz)'], subset['Storage Modulus (MPa)'], 'o', markersize=4, color=color, label=f'{ref_temp}°C' if temp == min(temperatures) else "")

        # Perform curve fitting
        combined_log_freq = np.array(combined_log_freq)
        combined_storage_modulus = np.array(combined_storage_modulus)
        params, _ = curve_fit(storage_modulus_model, 
                    combined_log_freq, 
                    combined_storage_modulus, 
                    bounds=(lower_bounds, upper_bounds), 
                    maxfev=10000
        )
        
        # Extract the estimated parameters
        a, b, c, d = params
        abcd_parameters.append((ref_temp, a, b, c, d))  # Append parameters with reference temperature
        # Plotting the fitted curve
        fitted_storage_modulus = storage_modulus_model(combined_log_freq, *params)
        plt.plot(10**combined_log_freq, fitted_storage_modulus, color='black', linestyle='--', linewidth=1)
        
        r2 = r2_score(combined_storage_modulus, fitted_storage_modulus)
        r2_scores[ref_temp] = r2
        
        # Plotting the fitted curve
        fitted_storage_modulus = storage_modulus_model(combined_log_freq, *params)
        plt.plot(10**combined_log_freq, fitted_storage_modulus, color='black', linestyle='--', linewidth=1)

        # Create a DataFrame for the current reference temperature
        df = pd.DataFrame({
            'Reference Temperature (°C)': [ref_temp] * len(combined_log_freq),
            'Reduced Frequency (Hz)': 10**combined_log_freq,
            'Storage Modulus (MPa)': combined_storage_modulus,
            'Fitted Storage Modulus (MPa)': fitted_storage_modulus
        })
        all_data_frames.append(df)

    # Adding labels, title, grid, and legend
    plt.xlabel('Reduced Frequency (Hz)', fontsize=40)
    plt.ylabel(r"$\it{E}'$ (MPa)", fontsize=40,fontname='Times New Roman')
    plt.xlim(1e-12, 1e12)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(fontsize=20)

    output_path = os.path.join(output_dir, 'plot9.png')  # Define the full path
    plt.savefig(output_path)
    plt.close()

    ######################
    ######################    

    for params in abcd_parameters:
        ref_temp, a, b, c, d = params
        print(f"Ref Temp {ref_temp}°C - A: {a}, B: {b}, C: {c}, D: {d}")

    # Print the R^2 scores
    for ref_temp, r2 in r2_scores.items():
        print(f"Ref Temp {ref_temp}°C: R² = {r2:.9f}")

    # # Define the function E'(w)
    def E_prime_param(w, a, b, c, d):
        return a * np.tanh(b * ((np.log(w)) + c)) + d

    # Function time_cycle for Etime
    def Etime_time_cycle_all(time, cycle, a, b, c, d):
        # Integration of sigmoidal curve numerically to get E(t)
        # Number of Sample Points per cycle
        N1, N2, N3 = 240, 74, 24

        # Initialize Etime
        Etime = np.zeros_like(time)

        # Define integrand function
        def integrand(t, E_prime_w, w):
            return (2/np.pi)*(E_prime_w/w)*np.sin(w*t)

        # Process each time point
        for i, t in enumerate(time):
            # Conditions for different ranges of cycles
            w1 = np.linspace((1e-6 / t), (cycle * 0.1 * 2 * np.pi / t), int(cycle * 0.1 * N1 + 1))
            w2 = np.linspace((cycle * 0.1 * 2 * np.pi) / t, (cycle * 0.4 * 2 * np.pi) / t, int(cycle * 0.3 * N2 + 1))
            w3 = np.linspace((cycle * 0.4 * 2 * np.pi) / t, (cycle * 2 * np.pi) / t, int(cycle * 0.6 * N3 + 1))

            # Concatenating all frequencies
            all_w = np.concatenate([w1, w2[1:], w3[1:]])  # Avoid repeating points at boundaries

            # Compute integrand for all frequencies
            y = integrand(t, E_prime_param(all_w, a, b, c, d), all_w)

            # Integrate using the trapezoidal method
            Etime[i] = np.trapezoid(y, all_w)

        return Etime
    
    ######################
    ######################

    ### E'(w) for each Reference Temp  ###

    # Creating a new figure for the plot
    plt.figure(figsize=(12, 8))

    # Loop over each set of parameters and plot E'(w)
    for params in abcd_parameters:
        ref_temp, A, B, C, D = params
        E_prime_values = E_prime_param(time, A, B, C, D)
        plt.plot(time, E_prime_values, label=f"Ref Temp {ref_temp}°C")

    # Plot settings
    plt.xlabel('Frequency (w) (Hz)')
    plt.ylabel("E'(w) (MPa)")
    plt.title("Plot of E'(w) at Different Reference Temperatures")
    plt.xscale('log')  # Logarithmic scale for frequency
    plt.grid(True)
    plt.legend()

    output_path = os.path.join(output_dir, 'plot11.png')  # Define the full path
    plt.savefig(output_path)
    plt.close()

    ######################
    ######################

    # Calculate the stress at the strain limit for each strain rate and ref temp
    elastic_modulus_dict = {temp: [] for temp in sorted(estimated_shift_factors.keys())}
    for params in abcd_parameters:
        ref_temp, A, B, C, D = params
        for rate in strain_rates:
            # Time corresponding to the strain limit for the current strain rate
            time_at_limit = strain_limit / rate

            # Compute E(t) at this time
            Etime_at_limit = Etime_time_cycle_all(np.array([time_at_limit]), 500, A, B, C, D)

            # The stress at the strain limit is E(t) * strain
            stress_at_limit = Etime_at_limit[0] * strain_limit

            # Elastic modulus is stress/strain
            modulus = stress_at_limit / strain_limit
            elastic_modulus_dict[ref_temp].append(modulus)

    # Plotting Elastic modulus vs temperature for each strain rate
    plt.figure(figsize=(14, 10))

    # Define colormap for different strain rates
    colors = cm.viridis(np.linspace(0, 1, len(strain_rates)))

    for i, rate in enumerate(strain_rates):
        moduli = [elastic_modulus_dict[temp][i] for temp in sorted(elastic_modulus_dict.keys())]
        plt.plot(sorted(elastic_modulus_dict.keys()), moduli, 'o-',  markersize=3, color=colors[i], label=f'Strain rate {rate} 1/s')

    plt.xlabel('Temperature (°C)', fontsize=40)
    plt.ylabel('Elastic Modulus (MPa)', fontsize=40)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(fontsize=18)
    plt.xlim(30, )

    output_path = os.path.join(output_dir, 'plot15.png')  # Define the full path
    plt.savefig(output_path)
    plt.close()

    ######################
    ######################

    ### Plot and Save Plot Etime, Elastic modulus at different strain rate and Ref_temperature ###

    # Initialize a DataFrame to store E(t) data for all reference temperatures
    etime_df = pd.DataFrame({'Time (s)': time})

    # Plotting Etime vs time for each reference temperature
    plt.figure(figsize=(12, 8))

    # Looping over each set of abcd parameters
    for params in abcd_parameters:
        ref_temp, A, B, C, D = params
        Etime = Etime_time_cycle_all(time, 500, A, B, C, D)
        plt.plot(time, Etime, label=f'Ref Temp {ref_temp}°C')
        etime_df[f'Ref Temp {ref_temp}°C'] = Etime

    plt.xlabel('Time (s)', fontsize=24)
    plt.ylabel('Relaxation Modulus (MPa)', fontsize=24)
    plt.title('Relaxation Modulus with Time at Different Reference Temperatures', fontsize=24)
    plt.xscale('log')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(time[0], time[-1])
    plt.legend(fontsize=12)

    output_path = os.path.join(output_dir, 'plot15.png')  # Define the full path
    plt.savefig(output_path)
    plt.close()

    ######################
    ######################

    # Define Secant Modulus and strain rates
    strain_limit = 0.25 / 100   # 0.25% Secant Modulus
    strain_rates = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
    # strain_limit = 0.25 / 100   # 0.25% Secant Modulus
    # strain_rates = [1e-6,1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]

    # Initialize a dictionary to store elastic modulus for each ref temp
    elastic_modulus_dict = {temp: [] for temp in sorted(estimated_shift_factors.keys())}

    # Calculate the stress at the strain limit for each strain rate and ref temp
    for params in abcd_parameters:
        ref_temp, A, B, C, D = params
        for rate in strain_rates:
            # Time corresponding to the strain limit for the current strain rate
            time_at_limit = strain_limit / rate

            # Compute E(t) at this time
            Etime_at_limit = Etime_time_cycle_all(np.array([time_at_limit]), 500, A, B, C, D)

            # The stress at the strain limit is E(t) * strain
            stress_at_limit = Etime_at_limit[0] * strain_limit

            # Elastic modulus is stress/strain
            modulus = stress_at_limit / strain_limit
            elastic_modulus_dict[ref_temp].append(modulus)

    # Plotting Elastic modulus vs strain rate for each Ref temp
    plt.figure(figsize=(12, 8))
    for temp, modulus_values in elastic_modulus_dict.items():
        plt.plot(strain_rates, modulus_values, marker='o', markersize=3, label=f'Ref Temp {temp}°C')

    plt.xlabel('Strain Rate (1/s)', fontsize=24)
    plt.ylabel('Elastic Modulus (MPa)', fontsize=24)
    plt.title(f'Elastic Modulus at {strain_limit*100}% Secant Modulus ', fontsize=24)
    plt.xscale('log')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(strain_rates[0], strain_rates[-1])
    plt.legend(fontsize=12)

    output_path = os.path.join(output_dir, 'plot16.png')  # Define the full path
    plt.savefig(output_path)
    plt.close()

    ######################
    ######################

    # Plotting Elastic modulus vs temperature for each strain rate
    plt.figure(figsize=(12, 8))

    # Define colormap for different strain rates
    colors = cm.viridis(np.linspace(0, 1, len(strain_rates)))

    for i, rate in enumerate(strain_rates):
        moduli = [elastic_modulus_dict[temp][i] for temp in sorted(elastic_modulus_dict.keys())]
        plt.plot(sorted(elastic_modulus_dict.keys()), moduli, 'o-', markersize=3, color=colors[i], label=f'Strain rate {rate} 1/s')

    plt.xlabel('Temperature (°C)', fontsize=24)
    plt.ylabel('Elastic Modulus (MPa)', fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title('Elastic Modulus vs Temperature for Different Strain Rates', fontsize=24)
    plt.legend(fontsize=12)

    output_path = os.path.join(output_dir, 'plot17.png')  # Define the full path
    plt.savefig(output_path)
    plt.close()

    ######################
    ######################

    ### Plot stress strain curve ###

    # Parameters for E'(w) when T_ref = 30
    A= 25.275332767062764
    B= 0.12575005134052475
    C= 9.959198426666351
    D= 55.41812750308131

    # Function to compute the stress-strain curve
    def stress_strain_ann(strain_rates):
        strain_limit = 0.25 / 100  # ~= Secant Modulus
        step_num = 501

        strain = np.linspace(1e-10, strain_limit, step_num)
        stress = np.zeros((step_num, len(strain_rates)))

        for i in range(len(strain_rates)):
            time_history = strain / strain_rates[i]
            Etime = Etime_time_cycle(time_history)
            for j in range(step_num):
                stress[j, i] = np.trapezoid(Etime[:j+1], strain[:j+1])

        stress_strain_curve = np.zeros((len(strain) + 1, len(strain_rates) + 1))
        stress_strain_curve[0, 1:] = strain_rates
        stress_strain_curve[1:, 0] = strain
        stress_strain_curve[1:, 1:] = stress

        return stress_strain_curve

    # Define some strain rates to plot
    strain_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]

    # Compute the stress-strain curve
    stress_strain_curve = stress_strain_ann(strain_rates)

    # Plotting the stress-strain curve
    plt.figure(figsize=(12, 8))
    for i in range(len(strain_rates)):
        plt.plot(stress_strain_curve[1:, 0], stress_strain_curve[1:, i+1], label=f'Strain Rate: {strain_rates[i]} 1/s')

    plt.xlabel('Strain')
    plt.ylabel('Stress (MPa)')
    plt.title('Stress-Strain Curve for Different Strain Rates')
    plt.legend()
    plt.grid(True)

    output_path = os.path.join(output_dir, 'plot18.png')  # Define the full path
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python converted_script.py <csv_file_path> <shift_factors> <use_equation>")
        sys.exit(1)

    csv_file_path = sys.argv[1]
    shift_factors = sys.argv[2]
    use_equation = sys.argv[3].lower() == 'true'
    process_csv(csv_file_path, shift_factors, use_equation), use_equation