import json
from datetime import datetime, timedelta

# Generate demo predictions for next 7 days
demo_predictions = []
base_price = 24150.5
start_date = datetime.now()

for i in range(7):
    next_date = start_date + timedelta(days=i+1)
    # Skip weekends
    while next_date.weekday() >= 5:
        next_date += timedelta(days=1)
    
    # Add some variance
    price_change = (-1)**i * (i * 25) + (i * 10)
    price = base_price + price_change
    
    demo_predictions.append({
        "Date": next_date.strftime("%Y-%m-%d"),
        "Price": round(price, 2)
    })

# Save to JSON
with open('artifacts/demo_predictions.json', 'w') as f:
    json.dump(demo_predictions, f, indent=2)

print("Demo predictions created:")
print(json.dumps(demo_predictions, indent=2))
