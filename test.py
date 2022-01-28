from train import sales_prediction

sls = sales_prediction()
df = sls.get_data()
df1 = sls.drop_irrelevent_info(df=df)
print(df1)
l = sls.get_x_and_y(df=df1)
x = l[0]
y = l[1]
m = sls.get_scaling(x=x)
scalar = m[1]
x=m[0]
print(x)
print(y)
l = [[230.1,37.8,69.2]]
print(l)
print(sls.get_prediction(l))

