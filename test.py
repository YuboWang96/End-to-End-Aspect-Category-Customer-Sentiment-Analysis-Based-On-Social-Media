import analysis_model 

a = analysis_model.restaurant_review("Nice and cozy sushi restaurant. Rich miso soup was good. Rhay had kind service.")
print(a.to_dict(orient='records'))

