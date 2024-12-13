#After training, we'll evaluate the model's performance and use it for predictions
# Load the model
best_model = tf.keras.models.load_model('best_model.h5')

# Evaluate on test data
test_loss = best_model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")

# Predicting future sales
future_dates = pd.date_range(start=df['date'].max(), periods=30)  # Predict next 30 days
future_forecast = []

# Assuming we have real-time data for the last 10 days to keep predicting
last_sequence = scaled_features[-seq_length:]
for _ in range(30):
    next_pred = best_model.predict(last_sequence.reshape(1, seq_length, len(features)))[0][0]
    future_forecast.append(scaler_y.inverse_transform([[next_pred]])[0][0])
    new_seq = np.append(last_sequence[1:], scaler_x.transform([[next_pred] + [0] * (len(features) - 1)]), axis=0)
    last_sequence = new_seq

# Convert predictions back to original scale
future_forecast_df = pd.DataFrame({
    'date': future_dates,
    'forecasted_sales': future_forecast
})

print(future_forecast_df)
