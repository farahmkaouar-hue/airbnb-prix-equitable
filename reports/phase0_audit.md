# Phase 0 : audit des données brutes

## Annonces
- shape : 485,128 lignes x 94 colonnes
- mémoire (pandas) : 1,487 Mo
- doublons sur `id` : 0 ; doublons (city, id) : 0

### Types
|         |   nb_colonnes |
|:--------|--------------:|
| float64 |            41 |
| str     |            33 |
| int64   |            20 |

### Dtype par colonne
|       | city   | country   | scrape_date   | id    | listing_url   | scrape_id   | last_scraped   | source   | name   | description   | neighborhood_overview   | picture_url   | host_id   | host_url   | host_profile_id   | host_profile_url   | host_name   | host_since   | hosts_time_as_user_years   | hosts_time_as_user_months   | hosts_time_as_host_years   | hosts_time_as_host_months   | host_location   | host_about   | host_response_time   | host_response_rate   | host_acceptance_rate   | host_is_superhost   | host_thumbnail_url   | host_picture_url   | host_neighbourhood   | host_listings_count   | host_total_listings_count   | host_verifications   | host_has_profile_pic   | host_identity_verified   | neighbourhood   | neighbourhood_cleansed   | neighbourhood_group_cleansed   | latitude   | longitude   | property_type   | room_type   | accommodates   | bathrooms   | bathrooms_text   | bedrooms   | beds    | amenities   | price   | price_quote_checkin_date   | price_quote_checkout_date   | price_quote_total_price   | price_quote_price_per_night   | price_quote_raw   | minimum_nights   | maximum_nights   | minimum_minimum_nights   | maximum_minimum_nights   | minimum_maximum_nights   | maximum_maximum_nights   | minimum_nights_avg_ntm   | maximum_nights_avg_ntm   | calendar_updated   | has_availability   | availability_30   | availability_60   | availability_90   | availability_365   | calendar_last_scraped   | number_of_reviews   | number_of_reviews_ltm   | number_of_reviews_l30d   | availability_eoy   | number_of_reviews_ly   | estimated_occupancy_l365d   | estimated_revenue_l365d   | first_review   | last_review   | review_scores_rating   | review_scores_accuracy   | review_scores_cleanliness   | review_scores_checkin   | review_scores_communication   | review_scores_location   | review_scores_value   | license   | instant_bookable   | calculated_host_listings_count   | calculated_host_listings_count_entire_homes   | calculated_host_listings_count_private_rooms   | calculated_host_listings_count_shared_rooms   | reviews_per_month   | currency   |
|:------|:-------|:----------|:--------------|:------|:--------------|:------------|:---------------|:---------|:-------|:--------------|:------------------------|:--------------|:----------|:-----------|:------------------|:-------------------|:------------|:-------------|:---------------------------|:----------------------------|:---------------------------|:----------------------------|:----------------|:-------------|:---------------------|:---------------------|:-----------------------|:--------------------|:---------------------|:-------------------|:---------------------|:----------------------|:----------------------------|:---------------------|:-----------------------|:-------------------------|:----------------|:-------------------------|:-------------------------------|:-----------|:------------|:----------------|:------------|:---------------|:------------|:-----------------|:-----------|:--------|:------------|:--------|:---------------------------|:----------------------------|:--------------------------|:------------------------------|:------------------|:-----------------|:-----------------|:-------------------------|:-------------------------|:-------------------------|:-------------------------|:-------------------------|:-------------------------|:-------------------|:-------------------|:------------------|:------------------|:------------------|:-------------------|:------------------------|:--------------------|:------------------------|:-------------------------|:-------------------|:-----------------------|:----------------------------|:--------------------------|:---------------|:--------------|:-----------------------|:-------------------------|:----------------------------|:------------------------|:------------------------------|:-------------------------|:----------------------|:----------|:-------------------|:---------------------------------|:----------------------------------------------|:-----------------------------------------------|:----------------------------------------------|:--------------------|:-----------|
| dtype | str    | str       | str           | int64 | str           | int64       | str            | str      | str    | str           | float64                 | str           | int64     | str        | int64             | str                | str         | float64      | float64                    | float64                     | float64                    | float64                     | str             | str          | float64              | float64              | float64                | str                 | float64              | str                | float64              | int64                 | float64                     | float64              | str                    | str                      | float64         | str                      | str                            | float64    | float64     | str             | str         | int64          | float64     | str              | float64    | float64 | str         | float64 | str                        | str                         | float64                   | float64                       | str               | float64          | float64          | float64                  | float64                  | float64                  | float64                  | float64                  | float64                  | float64            | str                | int64             | int64             | int64             | int64              | str                     | int64               | int64                   | int64                    | int64              | int64                  | int64                       | float64                   | str            | str           | float64                | float64                  | float64                     | float64                 | float64                       | float64                  | float64               | str       | float64            | int64                            | int64                                         | int64                                          | int64                                         | float64             | str        |

### Dates de collecte
- plage : 2026-09-14 → 2026-09-25
- nb de dates distinctes par ville : 1 (max)

### Annonces par ville
| city          | country         | devise   |   annonces | date       |
|:--------------|:----------------|:---------|-----------:|:-----------|
| london        | united-kingdom  | GBP      |      91954 | 2026-09-17 |
| paris         | france          | EUR      |      77149 | 2026-09-16 |
| los-angeles   | united-states   | USD      |      43781 | 2026-09-14 |
| rome          | italy           | EUR      |      37451 | 2026-09-17 |
| tokyo         | japan           | JPY      |      35970 | 2026-09-25 |
| mexico-city   | mexico          | MXN      |      30468 | 2026-09-23 |
| new-york-city | united-states   | USD      |      29918 | 2026-09-14 |
| lisbon        | portugal        | EUR      |      25020 | 2026-09-19 |
| madrid        | spain           | EUR      |      23109 | 2026-09-17 |
| toronto       | canada          | CAD      |      22050 | 2026-09-16 |
| sydney        | australia       | AUD      |      20833 | 2026-09-16 |
| barcelona     | spain           | EUR      |      15236 | 2026-09-17 |
| berlin        | germany         | EUR      |      12376 | 2026-09-19 |
| amsterdam     | the-netherlands | EUR      |      10612 | 2026-09-16 |
| lyon          | france          | EUR      |       9201 | 2026-09-17 |

### Colonnes 100 % vides par ville
- amsterdam : ['neighborhood_overview', 'host_since', 'host_response_time', 'host_response_rate', 'host_acceptance_rate', 'host_thumbnail_url', 'host_neighbourhood', 'host_total_listings_count', 'host_verifications', 'neighbourhood', 'neighbourhood_group_cleansed', 'bathrooms', 'calendar_updated', 'instant_bookable']
- barcelona : ['neighborhood_overview', 'host_since', 'host_response_time', 'host_response_rate', 'host_acceptance_rate', 'host_thumbnail_url', 'host_neighbourhood', 'host_total_listings_count', 'host_verifications', 'neighbourhood', 'bathrooms', 'calendar_updated', 'instant_bookable']
- berlin : ['neighborhood_overview', 'host_since', 'host_response_time', 'host_response_rate', 'host_acceptance_rate', 'host_thumbnail_url', 'host_neighbourhood', 'host_total_listings_count', 'host_verifications', 'neighbourhood', 'bathrooms', 'calendar_updated', 'instant_bookable']
- lisbon : ['neighborhood_overview', 'host_since', 'host_response_time', 'host_response_rate', 'host_acceptance_rate', 'host_thumbnail_url', 'host_neighbourhood', 'host_total_listings_count', 'host_verifications', 'neighbourhood', 'bathrooms', 'calendar_updated', 'instant_bookable']
- london : ['neighborhood_overview', 'host_since', 'host_response_time', 'host_response_rate', 'host_acceptance_rate', 'host_thumbnail_url', 'host_neighbourhood', 'host_total_listings_count', 'host_verifications', 'neighbourhood', 'neighbourhood_group_cleansed', 'bathrooms', 'calendar_updated', 'license', 'instant_bookable']
- los-angeles : ['neighborhood_overview', 'host_since', 'host_response_time', 'host_response_rate', 'host_acceptance_rate', 'host_thumbnail_url', 'host_neighbourhood', 'host_total_listings_count', 'host_verifications', 'neighbourhood', 'bathrooms', 'calendar_updated', 'instant_bookable']
- lyon : ['neighborhood_overview', 'host_since', 'host_response_time', 'host_response_rate', 'host_acceptance_rate', 'host_thumbnail_url', 'host_neighbourhood', 'host_total_listings_count', 'host_verifications', 'neighbourhood', 'neighbourhood_group_cleansed', 'bathrooms', 'calendar_updated', 'instant_bookable']
- madrid : ['neighborhood_overview', 'host_since', 'host_response_time', 'host_response_rate', 'host_acceptance_rate', 'host_thumbnail_url', 'host_neighbourhood', 'host_total_listings_count', 'host_verifications', 'neighbourhood', 'bathrooms', 'calendar_updated', 'instant_bookable']
- mexico-city : ['neighborhood_overview', 'host_since', 'host_response_time', 'host_response_rate', 'host_acceptance_rate', 'host_thumbnail_url', 'host_neighbourhood', 'host_total_listings_count', 'host_verifications', 'neighbourhood', 'neighbourhood_group_cleansed', 'bathrooms', 'calendar_updated', 'license', 'instant_bookable']
- new-york-city : ['neighborhood_overview', 'host_since', 'host_response_time', 'host_response_rate', 'host_acceptance_rate', 'host_thumbnail_url', 'host_neighbourhood', 'host_total_listings_count', 'host_verifications', 'neighbourhood', 'bathrooms', 'calendar_updated', 'instant_bookable']
- paris : ['neighborhood_overview', 'host_since', 'host_response_time', 'host_response_rate', 'host_acceptance_rate', 'host_thumbnail_url', 'host_neighbourhood', 'host_total_listings_count', 'host_verifications', 'neighbourhood', 'neighbourhood_group_cleansed', 'bathrooms', 'calendar_updated', 'instant_bookable']
- rome : ['neighborhood_overview', 'host_since', 'host_response_time', 'host_response_rate', 'host_acceptance_rate', 'host_thumbnail_url', 'host_neighbourhood', 'host_total_listings_count', 'host_verifications', 'neighbourhood', 'neighbourhood_group_cleansed', 'bathrooms', 'calendar_updated', 'instant_bookable']
- sydney : ['neighborhood_overview', 'host_since', 'host_response_time', 'host_response_rate', 'host_acceptance_rate', 'host_thumbnail_url', 'host_neighbourhood', 'host_total_listings_count', 'host_verifications', 'neighbourhood', 'neighbourhood_group_cleansed', 'bathrooms', 'calendar_updated', 'instant_bookable']
- tokyo : ['neighborhood_overview', 'host_since', 'host_response_time', 'host_response_rate', 'host_acceptance_rate', 'host_thumbnail_url', 'host_neighbourhood', 'host_total_listings_count', 'host_verifications', 'neighbourhood', 'neighbourhood_group_cleansed', 'bathrooms', 'calendar_updated', 'instant_bookable']
- toronto : ['neighborhood_overview', 'host_since', 'host_response_time', 'host_response_rate', 'host_acceptance_rate', 'host_thumbnail_url', 'host_neighbourhood', 'host_total_listings_count', 'host_verifications', 'neighbourhood', 'neighbourhood_group_cleansed', 'bathrooms', 'calendar_updated', 'instant_bookable']

### Colonnes les plus manquantes (global)
|                              |   % manquant |
|:-----------------------------|-------------:|
| neighborhood_overview        |        100   |
| host_acceptance_rate         |        100   |
| host_response_rate           |        100   |
| host_response_time           |        100   |
| host_thumbnail_url           |        100   |
| host_since                   |        100   |
| neighbourhood                |        100   |
| instant_bookable             |        100   |
| calendar_updated             |        100   |
| bathrooms                    |        100   |
| host_total_listings_count    |        100   |
| host_verifications           |        100   |
| host_neighbourhood           |        100   |
| source                       |         72.7 |
| neighbourhood_group_cleansed |         69.2 |
| license                      |         51.1 |
| host_about                   |         45.6 |

### Échantillon colonnes sensibles
|    |   price | currency   | host_is_superhost   |   host_response_rate | bathrooms_text   | amenities                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
|---:|--------:|:-----------|:--------------------|---------------------:|:-----------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  0 |      97 | EUR        | t                   |                  nan | 1 shared bath    | ["Bed linens", "Hair dryer", "Iron", "Heating", "Smoke alarm", "Private entrance", "Cleaning products", "Lock on bedroom door", "Refrigerator", "Host greets you", "Carbon monoxide alarm", "Fire extinguisher", "Wifi", "Shampoo", "Essentials", "Hot water", "Coffee maker", "Hangers"]                                                                                                                                                                                                                                             |
|  1 |     nan | EUR        | t                   |                  nan | 1 shared bath    | ["Bed linens", "Shower gel", "Hair dryer", "Iron", "Heating", "Smoke alarm", "Private entrance", "Lock on bedroom door", "Refrigerator", "Host greets you", "Carbon monoxide alarm", "Fire extinguisher", "Wifi", "Shampoo", "Essentials", "Hot water", "Coffee maker", "Hangers"]                                                                                                                                                                                                                                                    |
|  2 |     nan | EUR        | f                   |                  nan | 1.5 baths        | ["Bed linens", "Smoke alarm", "Cooking basics", "Patio or balcony", "Luggage dropoff allowed", "Dishes and silverware", "Private entrance", "Paid parking on premises", "Coffee maker", "Stove", "Dishwasher", "Air conditioning", "Extra pillows and blankets", "Hangers", "Microwave", "TV with standard cable", "Heating", "Refrigerator", "Carbon monoxide alarm", "Shampoo", "Oven", "Essentials", "Hot water", "Host greets you", "Shower gel", "Hair dryer", "Iron", "Kitchen", "Paid parking off premises", "Wifi", "Washer"] |

### host_is_superhost
| host_is_superhost   |   count |
|:--------------------|--------:|
| f                   |  339725 |
| t                   |  145403 |

## Avis
- 21,726,192 lignes x 8 colonnes (fichier 2.60 Go)
- plage des dates d'avis : 2009-05-25 → 2026-10-03

### Avis par ville
| city          |    avis |
|:--------------|--------:|
| amsterdam     |  571927 |
| barcelona     | 1127248 |
| berlin        |  708768 |
| lisbon        | 1950076 |
| london        | 2323969 |
| los-angeles   | 1890200 |
| lyon          |  473321 |
| madrid        | 1524765 |
| mexico-city   | 1795294 |
| new-york-city |  999099 |
| paris         | 2503875 |
| rome          | 2622221 |
| sydney        |  829163 |
| tokyo         | 1677064 |
| toronto       |  729202 |

## Calendar
- fichiers calendar trouvés dans data/raw : 0