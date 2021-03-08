#%%
import urllib.request

from hedgiefinder import predict
#%% source grabbed from instagram
file = r"https://scontent-msp1-1.cdninstagram.com/v/t50.2886-16/158270992_453068886137368_3067854260702542789_n.mp4?_nc_ht=scontent-msp1-1.cdninstagram.com\u0026_nc_cat=104\u0026_nc_ohc=KcldJnHoYbAAX-LYe4v\u0026oe=60474255\u0026oh=382ed10754684cfa809426f349b7fd34"
file.replace('u0026', '&')

# file = "https://scontent-msp1-1.cdninstagram.com/v/t50.2886-16/158270992_453068886137368_3067854260702542789_n.mp4?_nc_ht=scontent-msp1-1.cdninstagram.com&_nc_cat=104&_nc_ohc=KcldJnHoYbAAX-LYe4v&oe=60474255&oh=382ed10754684cfa809426f349b7fd34"

# %% download
urllib.request.urlretrieve(file, 'my_test_vid.mp4')

# %% predict
predict('my_test_vid.mp4')