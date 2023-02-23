import dataframe
import sqrt_distance_classifier

df = dataframe.Dataframe('test-data/Hotel Reservations.csv')
model = sqrt_distance_classifier.train(df, ['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights', 'room_type_reserved', 'lead_time', 'market_segment_type', 'repeated_guest', 'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled', 'avg_price_per_room', 'no_of_special_requests'])
print(sqrt_distance_classifier.eval(model, df, 'booking_status'))