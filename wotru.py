import pymongo
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import streamlit as st
import pydeck as pdk
from datetime import datetime
import matplotlib.pyplot as plt

# MongoDB connection
MONGO_URL = "mongodb+srv://upendrataral21:safety123@cluster0.x0pyhch.mongodb.net/avert?retryWrites=true&w=majority"
COLLECTION_NAME = "avert_data"
connection = pymongo.MongoClient(MONGO_URL)

database = connection.get_database()
collection = database[COLLECTION_NAME]
st.header('3D Visualization of Location Clusters')


def fetch_data():
    all_data = collection.find()
    sos_true_data_points = []  # Locations of people with SOS set to True
    for data in all_data:
        try:
            latitude = float(data["latitude"])
            longitude = float(data["longitude"])
            sos = data["sos"]
            if sos.lower() == 'true':  # Check if SOS is True
                sos_true_data_points.append((latitude, longitude))
        except KeyError:
            st.warning("")
    return sos_true_data_points


def plot_maps(sos_true_data_points):
    if len(sos_true_data_points) > 150:
        df_sos_true = pd.DataFrame(sos_true_data_points, columns=["latitude", "longitude"])
        st.subheader("Locations of people with SOS set to True")
        # st.pydeck_chart(pdk.Deck(
        #     map_style="mapbox://styles/mapbox/light-v9",
        #     initial_view_state=pdk.ViewState(
        #         latitude=df_sos_true["latitude"].mean(),
        #         longitude=df_sos_true["longitude"].mean(),
        #         zoom=10,
        #         pitch=50,
        #     ),
        #     layers=[
        #         pdk.Layer(
        #             "HexagonLayer",
        #             data=df_sos_true,
        #             get_position="[longitude, latitude]",
        #             radius=100,
        #             elevation_scale=4,
        #             elevation_range=[0, 1000],
        #             pickable=True,
        #             extruded=True,
        #         ),
        #     ],
        # ))

        # Cluster the data using KMeans
        num_clusters = min(len(sos_true_data_points), 10)  # Limiting to 10 clusters at most
        kmeans = KMeans(n_clusters=num_clusters)
        X = np.array(df_sos_true)
        kmeans.fit(X)
        centroids = kmeans.cluster_centers_

        # Plotting the cluster centers and circumferences
        cluster_info = [{"latitude": centroid[0], "longitude": centroid[1]} for centroid in centroids]
        df_clusters = pd.DataFrame(cluster_info)
        cluster_layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_clusters,
            get_position="[longitude, latitude]",
            get_radius=500,  # Radius in meters (diameter of 1000 meters)
            get_fill_color=[255, 0, 0],  # Red color for cluster regions
            opacity=0.3,
        )
        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=pdk.ViewState(
                latitude=df_sos_true["latitude"].mean(),
                longitude=df_sos_true["longitude"].mean(),
                zoom=10,
                pitch=50,
            ),
            layers=[
                pdk.Layer(
                    "HexagonLayer",
                    data=df_sos_true,
                    get_position="[longitude, latitude]",
                    radius=100,
                    elevation_scale=4,
                    elevation_range=[0, 1000],
                    pickable=True,
                    extruded=True,
                ),
                cluster_layer,
            ],
        ))


def analytical_panel(sos_true_data_points):
    if sos_true_data_points:
        st.sidebar.subheader("Analytical Panel")
        st.sidebar.write("Total number of people with SOS set to True:", len(sos_true_data_points))
        #st.sidebar.write("Average SOS frequency:",  "Calculate average SOS frequency if available "Not available")

        # Pie chart for SOS distribution
        sos_distribution = pd.Series(["SOS True", "SOS False"]).value_counts()
        fig, ax = plt.subplots()
        ax.pie(sos_distribution, labels=sos_distribution.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.sidebar.subheader("SOS Distribution")
        st.sidebar.pyplot(fig)

        # Additional visual representation method
        # You can add your preferred visualization method here


def main():
    if st.button("Fetch Updated SOS Locations"):
        sos_true_data_points = fetch_data()
        st.write("Data Updated on : ", str(datetime.now()))
        plot_maps(sos_true_data_points)
        analytical_panel(sos_true_data_points)


if __name__ == "__main__":
    main()
