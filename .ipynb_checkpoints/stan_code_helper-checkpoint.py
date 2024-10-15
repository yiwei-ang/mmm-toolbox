def create_stan_function_block():
    return """
    functions {
        // Generic Hill function
        real hill_function(real x, real beta, real A, real B) {
            return beta * pow(x, A) / (pow(x + 1e-9, A) + pow(B, A));
        }
    }
    
    """

def create_stan_data_block(config_df: pd.DatFrame):
    spaces = "    "
    data_block_code = "data {\n"
    data_block_code += spaces + "int<lower=0> N; // Number of observations\n"
    
    # Iterate over rows in the configuration DataFrame, skipping non-media variables
    for _, row in config_df.iterrows():
        variable = row['Variable']
        
        category = row['Category']
        if category in ['MEDIA']:
            data_block_code += f"{spaces}vector[N] {variable}; // Media variable data\n"
    
        # Add base_sales and revenue if they are in the configuration
        if category in ['BASE']:
            data_block_code += spaces + f"vector[N] {variable}; // Base sales data\n"
        if category in ['KPI']:
            data_block_code += spaces + f"vector[N] revenue; // Target variable\n"
    
    data_block_code += "}\n"
    
    return data_block_code

# Function to create parameters block for Stan model
def create_stan_parameters_block(config_df):
    spaces = "    "
    parameters_code = "parameters {\n"
    
    # Iterate over rows in the configuration DataFrame
    for _, row in config_df.iterrows():
        variable = row['Variable']
        category = row['Category']
        
        # Skip the row if it's not a media variable or base revenue
        if category in ['KPI']:
            continue
        
        # Create beta, A, and B parameters for each media variable
        beta_name = f"beta_{variable}"
        A_name = f"param_A_{variable}"
        B_name = f"param_B_{variable}"
        
        parameters_code += f"{spaces}real <lower={row['Beta_Min']}, upper={row['Beta_Max']}> {beta_name}; // Coefficient for {variable}\n"
        
        if category in ["MEDIA"]:
            parameters_code += f"{spaces}real <lower={row['Parameter_A_Min']}, upper={row['Parameter_A_Max']}> {A_name}; // Hill parameter A for {variable}\n"
            parameters_code += f"{spaces}real <lower={row['Parameter_B_Min']}, upper={row['Parameter_B_Max']}> {B_name}; // Hill parameter B for {variable}\n"
    
    # # Add base revenue parameter
    # parameters_code += f"{spaces}real <lower=0.0> base_revenue; // Base revenue parameter\n"
    
    # # Add standard deviation of the residuals
    parameters_code += f"{spaces}real <lower=0.0> sigma; // Standard deviation of the residuals\n"
    
    parameters_code += "}\n"
    
    return parameters_code

def create_stan_model_block(config_df):
    spaces = "    "
    model_code = "model {\n"
    
    # Add priors for media variables
    for _, row in config_df.iterrows():
        variable = row['Variable']
        category = row['Category']
        
        if category == 'MEDIA':
            beta_name = f"beta_{variable}"
            A_name = f"param_A_{variable}"
            B_name = f"param_B_{variable}"
            
            # Assuming normal priors for all parameters for simplicity
            model_code += f"{spaces}{beta_name} ~ normal(0, {row['Prior']});\n"
            model_code += f"{spaces}{A_name} ~ normal(0, {row['Prior']});\n"
            model_code += f"{spaces}{B_name} ~ normal(0, {row['Prior']});\n"
    
    # Add priors for base sales
        if category == 'MEDIA':
            model_code += f"{spaces}{variable} ~ normal(0, {config_df[config_df['Variable'] == variable]['Prior'].values[0]});\n"

    # Add prior for the standard deviation of residuals
    model_code += f"{spaces}sigma ~ cauchy(0, 2);\n"
    
    # Define mu and add the likelihood using the Hill function for media variables and base sales
    model_code += f"{spaces}vector[N] mu = rep_vector(0, N);\n"
    model_code += f"{spaces}for (n in 1:N) {{\n"
    
    for _, row in config_df.iterrows():
        variable = row['Variable']
        category = row['Category']
        
        if category == 'MEDIA':
            beta_name = f"beta_{variable}"
            A_name = f"param_A_{variable}"
            B_name = f"param_B_{variable}"
            
            # Apply the Hill function to each media variable
            model_code += f"{spaces}{spaces}mu[n] += hill_function({variable}[n], {beta_name}, {A_name}, {B_name});\n"
    
        if category == 'BASE':
            model_code += f"{spaces}{spaces}mu[n] += {beta_name} * {variable}[n];\n"
    
    # Add the revenue likelihood
    model_code += f"{spaces}{spaces}revenue[n] ~ normal(mu[n], sigma);\n"
    model_code += f"{spaces}}}\n"
    
    model_code += "}\n"
    
    return model_code
