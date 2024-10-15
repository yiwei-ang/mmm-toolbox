
    functions {
        // Generic Hill function
        real hill_function(real x, real beta, real A, real B) {
            return beta * pow(x, A) / (pow(x + 1e-9, A) + pow(B, A));
        }
    }
    
    
data {
    int<lower=0> N; // Number of observations
    vector[N] revenue; // Target variable
    vector[N] base_revenue; // Base sales data
    vector[N] google_performance_max; // Media variable data
    vector[N] google_search_brand; // Media variable data
    vector[N] google_search_no_brand; // Media variable data
    vector[N] facebook_conversions; // Media variable data
    vector[N] facebook_others; // Media variable data
    vector[N] facebook_product_catalog_sales; // Media variable data
    vector[N] influencers; // Media variable data
    vector[N] display_ads; // Media variable data
    vector[N] tv_ads; // Media variable data
    vector[N] radio_ads; // Media variable data
    vector[N] magazine_ads; // Media variable data
}

parameters {
    real <lower=0.1, upper=1000.0> beta_base_revenue; // Coefficient for base_revenue
    real <lower=0.1, upper=1000.0> beta_google_performance_max; // Coefficient for google_performance_max
    real <lower=0.1, upper=2.0> param_A_google_performance_max; // Hill parameter A for google_performance_max
    real <lower=0.1, upper=1000.0> param_B_google_performance_max; // Hill parameter B for google_performance_max
    real <lower=0.1, upper=1000.0> beta_google_search_brand; // Coefficient for google_search_brand
    real <lower=0.1, upper=2.0> param_A_google_search_brand; // Hill parameter A for google_search_brand
    real <lower=0.1, upper=1000.0> param_B_google_search_brand; // Hill parameter B for google_search_brand
    real <lower=0.1, upper=1000.0> beta_google_search_no_brand; // Coefficient for google_search_no_brand
    real <lower=0.1, upper=2.0> param_A_google_search_no_brand; // Hill parameter A for google_search_no_brand
    real <lower=0.1, upper=1000.0> param_B_google_search_no_brand; // Hill parameter B for google_search_no_brand
    real <lower=0.1, upper=1000.0> beta_facebook_conversions; // Coefficient for facebook_conversions
    real <lower=0.1, upper=2.0> param_A_facebook_conversions; // Hill parameter A for facebook_conversions
    real <lower=0.1, upper=1000.0> param_B_facebook_conversions; // Hill parameter B for facebook_conversions
    real <lower=0.1, upper=1000.0> beta_facebook_others; // Coefficient for facebook_others
    real <lower=0.1, upper=2.0> param_A_facebook_others; // Hill parameter A for facebook_others
    real <lower=0.1, upper=1000.0> param_B_facebook_others; // Hill parameter B for facebook_others
    real <lower=0.1, upper=1000.0> beta_facebook_product_catalog_sales; // Coefficient for facebook_product_catalog_sales
    real <lower=0.1, upper=2.0> param_A_facebook_product_catalog_sales; // Hill parameter A for facebook_product_catalog_sales
    real <lower=0.1, upper=1000.0> param_B_facebook_product_catalog_sales; // Hill parameter B for facebook_product_catalog_sales
    real <lower=0.1, upper=1000.0> beta_influencers; // Coefficient for influencers
    real <lower=0.1, upper=2.0> param_A_influencers; // Hill parameter A for influencers
    real <lower=0.1, upper=1000.0> param_B_influencers; // Hill parameter B for influencers
    real <lower=0.1, upper=1000.0> beta_display_ads; // Coefficient for display_ads
    real <lower=0.1, upper=2.0> param_A_display_ads; // Hill parameter A for display_ads
    real <lower=0.1, upper=1000.0> param_B_display_ads; // Hill parameter B for display_ads
    real <lower=0.1, upper=1000.0> beta_tv_ads; // Coefficient for tv_ads
    real <lower=0.1, upper=2.0> param_A_tv_ads; // Hill parameter A for tv_ads
    real <lower=0.1, upper=1000.0> param_B_tv_ads; // Hill parameter B for tv_ads
    real <lower=0.1, upper=1000.0> beta_radio_ads; // Coefficient for radio_ads
    real <lower=0.1, upper=2.0> param_A_radio_ads; // Hill parameter A for radio_ads
    real <lower=0.1, upper=1000.0> param_B_radio_ads; // Hill parameter B for radio_ads
    real <lower=0.1, upper=1000.0> beta_magazine_ads; // Coefficient for magazine_ads
    real <lower=0.1, upper=2.0> param_A_magazine_ads; // Hill parameter A for magazine_ads
    real <lower=0.1, upper=1000.0> param_B_magazine_ads; // Hill parameter B for magazine_ads
    real <lower=0.0> sigma; // Standard deviation of the residuals
}

transformed parameters {
    vector[N] mu = rep_vector(0, N);
    for (n in 1:N) {
        mu[n] += beta_base_revenue * base_revenue[n];
        mu[n] += hill_function(google_performance_max[n], beta_google_performance_max, param_A_google_performance_max, param_B_google_performance_max);
        mu[n] += hill_function(google_search_brand[n], beta_google_search_brand, param_A_google_search_brand, param_B_google_search_brand);
        mu[n] += hill_function(google_search_no_brand[n], beta_google_search_no_brand, param_A_google_search_no_brand, param_B_google_search_no_brand);
        mu[n] += hill_function(facebook_conversions[n], beta_facebook_conversions, param_A_facebook_conversions, param_B_facebook_conversions);
        mu[n] += hill_function(facebook_others[n], beta_facebook_others, param_A_facebook_others, param_B_facebook_others);
        mu[n] += hill_function(facebook_product_catalog_sales[n], beta_facebook_product_catalog_sales, param_A_facebook_product_catalog_sales, param_B_facebook_product_catalog_sales);
        mu[n] += hill_function(influencers[n], beta_influencers, param_A_influencers, param_B_influencers);
        mu[n] += hill_function(display_ads[n], beta_display_ads, param_A_display_ads, param_B_display_ads);
        mu[n] += hill_function(tv_ads[n], beta_tv_ads, param_A_tv_ads, param_B_tv_ads);
        mu[n] += hill_function(radio_ads[n], beta_radio_ads, param_A_radio_ads, param_B_radio_ads);
        mu[n] += hill_function(magazine_ads[n], beta_magazine_ads, param_A_magazine_ads, param_B_magazine_ads);
    }
}

model {
    base_revenue ~ normal(1.0, 10000.0);
    beta_google_performance_max ~ normal(10.0, 10000.0);
    param_A_google_performance_max ~ normal(1.0, 10000.0);
    param_B_google_performance_max ~ normal(2.0, 10000.0);
    beta_google_search_brand ~ normal(10.0, 10000.0);
    param_A_google_search_brand ~ normal(1.0, 10000.0);
    param_B_google_search_brand ~ normal(2.0, 10000.0);
    beta_google_search_no_brand ~ normal(10.0, 10000.0);
    param_A_google_search_no_brand ~ normal(1.0, 10000.0);
    param_B_google_search_no_brand ~ normal(2.0, 10000.0);
    beta_facebook_conversions ~ normal(5.0, 10000.0);
    param_A_facebook_conversions ~ normal(1.0, 10000.0);
    param_B_facebook_conversions ~ normal(2.0, 10000.0);
    beta_facebook_others ~ normal(5.0, 10000.0);
    param_A_facebook_others ~ normal(1.0, 10000.0);
    param_B_facebook_others ~ normal(2.0, 10000.0);
    beta_facebook_product_catalog_sales ~ normal(5.0, 10000.0);
    param_A_facebook_product_catalog_sales ~ normal(1.0, 10000.0);
    param_B_facebook_product_catalog_sales ~ normal(5.0, 10000.0);
    beta_influencers ~ normal(5.0, 10000.0);
    param_A_influencers ~ normal(1.0, 10000.0);
    param_B_influencers ~ normal(5.0, 10000.0);
    beta_display_ads ~ normal(1.0, 10000.0);
    param_A_display_ads ~ normal(1.0, 10000.0);
    param_B_display_ads ~ normal(30.0, 10000.0);
    beta_tv_ads ~ normal(1.0, 10000.0);
    param_A_tv_ads ~ normal(1.0, 10000.0);
    param_B_tv_ads ~ normal(30.0, 10000.0);
    beta_radio_ads ~ normal(1.0, 10000.0);
    param_A_radio_ads ~ normal(1.0, 10000.0);
    param_B_radio_ads ~ normal(30.0, 10000.0);
    beta_magazine_ads ~ normal(1.0, 10000.0);
    param_A_magazine_ads ~ normal(1.0, 10000.0);
    param_B_magazine_ads ~ normal(30.0, 10000.0);
    sigma ~ cauchy(0, 2);
    for (n in 1:N) {
        revenue[n] ~ normal(mu[n], sigma);
    }
}
