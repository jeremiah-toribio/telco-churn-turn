# Telco Churn Turn
---
## Project Description (re)
Telco is a telecommunications company that would like to retrieve insight on whether there are relationships between customer churn and any type of service, plan or demographic. 
Here we will explore possible relationships within the customers' data to determine as such; it is important to note that coorelation is not directly causation and further determination may be required in order to make claims.
The customer data that we will be exploring will not include personally identifiable information and will only be used demographically.

## Project Goals
---
- Discover independence of features within customer data
- Utilizing said features to develop a machine learning model that will determine the likelyhood of the next customer to churn or not to churn.
- Churn is described here as 'a customer unsubscribing from services'
- The insights discovered will be used to make decisions to prevent further customer churn


## Initial Thoughts
--- 
When determining whether or not a customer will churn, my initial thought is there will likely be a few elements to that will not definitely determine if a customer will churn but may help in making a slightly better assumption that one will churn.

## Planning
--- 
1. Acquire data from MySQL Server
2. Prepare data accordingly for exploration & machine learning needs
    a. Creating dummy columns of multi categorical data
        - multiple_lines
        - online_security
        - online_backup
        - payment_type
        - contract_type
        - tech_support
        - streaming_tv
        - streaming_movies
        - device_protection
    b. Creating dummy columns of binary categorical data
        - partner
        - dependents
        - phone_service
        - gender
        - paperless_billing
        - churn
    c. Cleaning numeric data
        - total_charges
3. Explore data for assocations with churn
    a. Determine our baseline prediction
    b. Does the amount paid monthly determine likelihood to churn?
    c. Could the amount of services utilized determine churn?
    d. Will the customer contract basis determine churn?
    e. If the customer is single are they more likely to churn?
4. Develop a model to further understand churn
    a. Use various models to determine what algorithm are best to use
    b. Select best model based on evaluation metrics
    c. Evaluate all models on respective validate and test data

## Data Dictionary
--- 
| Feature        | Definition                                   |
| ---            | ---                                          |
| customer_id    | string; randomized 4 digit - 5 character identification code |
| gender         | string; male/female, determines account owner gender |
| senior_citizen | string; binary encoded, determines if account owner is a senior citizen |
| partner        | string; yes/no, determines if account owner has a partner |
| dependents     | string; yes/no, determines if account owner has any dependents |
| tenure         | integer; the number of years the customer has been with the company |
| phone_service  | string; yes/no/No phone service, multicategorical data to determine if the customer uses phone service |
| internet_service_type_id | int; 1/2/3, determines what type of internet service plan identifier the customer has |
| online_security| string; yes/no/No internet service, determines if the customer pays for the online security feature |
| online_backup  | string; yes/no determine if the customer has this feature |
| streaming_tv   | string; yes/no determine if the customer has this feature |
| streaming_movies | string; yes/no determine if the customer this feature |
| paperless_billing | string; yes/no determine if the customer has this feature|
| payment_type_id | integer; 1/2/3/4, determines what identifier for form of payment the customer utilizes |
| monthly_charges | integer; the amount the customer pays monthly |
| total_charges | integer; a total accumilation of what the customer has paid since beginning their service plan |
| churn | str; yes/no, determines whether or not a customer has churned, this is our target |
| internet_service_type | str; DSL/Fiber optic/No internet service, multicategorical data to determine what internet service type is used by the customer |
| payment_type | str; Mailed check/Electronic check/Credit card(automatic)/Credit card(manual), multicategorical data to determine the customer payment type in detail |
| contract_type | str; One year/Month-to-month/Two year, the type of contract type signed by the customer to determine length of service |
| Various encoded dummy columns of respective categorical types above |

## Reproducability Requirements
---
1. Clone repo
2. Establish credentials in *env.py* file in order to access codeup MySQL server
3. Run notebook

## Conclusions
---


## Recommendation
---