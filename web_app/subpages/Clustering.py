import os

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

from step_3_data_analysis.clustering import plot_sh_score, plot_k_means_on_pacmap, plot_k_prot_on_pacmap, \
    plot_sh_score_DBSCAN, plot_DBSCAN_on_pacmap, plot_SLINK_on_pacmap, plot_clusters_on_3D_pacmap, plot_sh_score_SLINK
from web_app.util import get_avg_cohort_cache, get_default_values, insert_feature_selectors


def my_plot_dendrogram(icoords, dcoords, ivl, p, n, mh, orientation, no_labels, color_list, leaf_font_size,
                       leaf_rotation, contraction_marks, ax, above_threshold_color):
    # Import matplotlib here so that it's not imported unless dendrograms
    # are plotted. Raise an informative error if importing fails.
    try:
        # if an axis is provided, don't use pylab at all
        if ax is None:
            import matplotlib.pylab
        import matplotlib.patches
        import matplotlib.collections
    except ImportError as e:
        raise ImportError("You must install the matplotlib library to plot "
                          "the dendrogram. Use no_plot=True to calculate the "
                          "dendrogram without plotting.") from e

    _dtextsizes = {20: 12, 30: 10, 50: 8, 85: 6, np.inf: 5}
    _dtextsortedkeys = list(_dtextsizes.keys())

    if ax is None:
        ax = matplotlib.pylab.gca()
        # if we're using pylab, we want to trigger a draw at the end
        trigger_redraw = True
    else:
        trigger_redraw = False

    # Independent variable plot width
    ivw = len(ivl) * 10
    # Dependent variable plot height
    dvw = mh + mh * 0.05

    iv_ticks = np.arange(5, len(ivl) * 10 + 5, 10)
    if orientation in ('top', 'bottom'):
        if orientation == 'top':
            ax.set_ylim([0, dvw])
            ax.set_xlim([0, ivw])
        else:
            ax.set_ylim([dvw, 0])
            ax.set_xlim([0, ivw])

        xlines = icoords
        ylines = dcoords
        if no_labels:
            ax.set_xticks([])
            ax.set_xticklabels([])
        else:
            ax.set_xticks(iv_ticks)

            if orientation == 'top':
                ax.xaxis.set_ticks_position('bottom')
            else:
                ax.xaxis.set_ticks_position('top')

            # Make the tick marks invisible because they cover up the links
            for line in ax.get_xticklines():
                line.set_visible(False)

            temp_tick_size = 12
            for k in _dtextsortedkeys:
                if len(ivl) <= k:
                    temp_tick_size = _dtextsizes[k]

            _drotation = {20: 0, 40: 45, np.inf: 90}
            _drotationsortedkeys = list(_drotation.keys())
            _drotationsortedkeys.sort()

            temp_tick_rotation = 0
            for k in _drotationsortedkeys:
                if p <= k:
                    temp_tick_rotation = _drotation[k]

            leaf_rot = (float(temp_tick_rotation)
                        if (leaf_rotation is None) else leaf_rotation)
            leaf_font = (float(temp_tick_size)
                         if (leaf_font_size is None) else leaf_font_size)
            ax.set_xticklabels(ivl, rotation=leaf_rot, size=leaf_font)

    elif orientation in ('left', 'right'):
        if orientation == 'left':
            ax.set_xlim([dvw, 0])
            ax.set_ylim([0, ivw])
        else:
            ax.set_xlim([0, dvw])
            ax.set_ylim([0, ivw])

        xlines = dcoords
        ylines = icoords
        if no_labels:
            ax.set_yticks([])
            ax.set_yticklabels([])
        else:
            ax.set_yticks(iv_ticks)

            if orientation == 'left':
                ax.yaxis.set_ticks_position('right')
            else:
                ax.yaxis.set_ticks_position('left')

            # Make the tick marks invisible because they cover up the links
            for line in ax.get_yticklines():
                line.set_visible(False)

            for k in _dtextsortedkeys:
                if len(ivl) <= k:
                    temp_tick_size = _dtextsizes[k]

            leaf_font = (float(temp_tick_size)
                         if (leaf_font_size is None) else leaf_font_size)

            if leaf_rotation is not None:
                ax.set_yticklabels(ivl, rotation=leaf_rotation, size=leaf_font)
            else:
                ax.set_yticklabels(ivl, size=leaf_font)

    # Let's use collections instead. This way there is a separate legend item
    # for each tree grouping, rather than stupidly one for each line segment.
    seen_before = set([])
    colors_used = []
    for i in color_list:
        if i not in seen_before:
            seen_before.add(i)
            colors_used.append(i)

    color_to_lines = {}
    for color in colors_used:
        color_to_lines[color] = []
    for (xline, yline, color) in zip(xlines, ylines, color_list):
        color_to_lines[color].append(list(zip(xline, yline)))

    colors_to_collections = {}
    # Construct the collections.
    for color in colors_used:
        coll = matplotlib.collections.LineCollection(color_to_lines[color],
                                                     colors=(color,))
        colors_to_collections[color] = coll

    # Add all the groupings below the color threshold.
    for color in colors_used:
        if color != above_threshold_color:
            ax.add_collection(colors_to_collections[color])
    # If there's a grouping of links above the color threshold, it goes last.
    if above_threshold_color in colors_to_collections:
        ax.add_collection(colors_to_collections[above_threshold_color])

    if contraction_marks is not None:
        Ellipse = matplotlib.patches.Ellipse
        for (x, y) in contraction_marks:
            if orientation in ('left', 'right'):
                e = Ellipse((y, x), width=dvw / 100, height=1.0)
            else:
                e = Ellipse((x, y), width=1.0, height=dvw / 100)
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(0.5)
            e.set_facecolor('k')

    if trigger_redraw:
        matplotlib.pylab.draw_if_interactive()


def clustering_page():
    ## Start of Page: User Input Selector
    st.markdown("<h1 style='text-align: left; color: black;'>Clustering</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns((0.25, 0.25, 0.25))
    ALL_DEPENDENT_VARIABLES: list = ['death_in_hosp', 'death_3_days', 'death_30_days', 'death_180_days',
                                     'death_365_days']
    selected_variable = col1.selectbox(label='Select dependent variable', options=ALL_DEPENDENT_VARIABLES)
    ALL_DATABASES: list = ['complete', 'metavision', 'carevue']
    selected_database = col2.selectbox(label='Select database', options=ALL_DATABASES)
    ALL_STROKE_TYPES: list = ['all_stroke', 'ischemic', 'other_stroke', 'hemorrhagic']
    selected_stroke_type = col3.selectbox(label='Select stroke type', options=ALL_STROKE_TYPES)
    cohort_title = 'scaled_' + selected_database + '_avg_cohort_' + selected_stroke_type

    ## Get Cohort from streamlit cache function
    upload_filename = './web_app/data_upload/exports/frontend/avg_patient_cohort.csv'
    if not os.path.isfile(upload_filename):
        st.warning('Warning: No dataset was uploaded. Please, first upload a dataset at the "Data Upload" page.')
    else:
        PROJECT_PATH = './web_app/data_upload/'
        FEATURES_DF = pd.read_excel('./supplements/FEATURE_PREPROCESSING_TABLE.xlsx')

        selected_cohort = get_avg_cohort_cache(project_path=PROJECT_PATH,
                                               use_case_name='frontend',
                                               features_df=FEATURES_DF,
                                               selected_database=selected_database,
                                               selected_stroke_type=selected_stroke_type,
                                               delete_existing_cache=False,
                                               selected_patients=[])  # empty = all
        # Feature Selector
        ALL_FEATURES = list(selected_cohort.columns)
        selected_features = insert_feature_selectors(ALL_FEATURES, ALL_DEPENDENT_VARIABLES, selected_variable)
        # st.markdown('___')

        ## Select Clustering Specific Parameters
        col5, col6, col7, col8 = st.columns((0.25, 0.25, 0.25, 0.25))
        ALL_CLUSTERING_METHODS: list = ['kmeans', 'kprototype', 'DBSCAN', 'SLINK']
        clustering_method = col5.selectbox(label='Select clustering method', options=ALL_CLUSTERING_METHODS)
        ALL_CRITERIA: list = ['maxclust', 'distance', 'inconsistent']     # 'monocrit',
        if clustering_method == 'kmeans' or clustering_method == 'kprototype':
            selected_cluster_count = col6.number_input(label='Select cluster count k', min_value=1, max_value=50, value=3)
            selected_eps = None
            selected_min_sample = None
            selected_criterion = None
            selected_threshold = None
        elif clustering_method == 'DBSCAN':
            selected_cluster_count = None
            selected_eps = col6.number_input(label='Select epsilon', min_value=0.01, max_value=10.00, value=0.51)
            selected_min_sample = col7.number_input(label='Select min_sample', min_value=1, max_value=100, value=5)
            selected_criterion = None
            selected_threshold = None
        elif clustering_method == 'SLINK':
            selected_cluster_count = None
            selected_eps = None
            selected_min_sample = None
            selected_criterion = col6.selectbox(label='Select separation criterion', options=ALL_CRITERIA)
            selected_threshold = col7.number_input(label='Select threshold t', min_value=0.01, max_value=100.00, value=5.00, step=1.00)
        else:
            selected_cluster_count = None
            selected_eps = None
            selected_min_sample = None
            selected_criterion = None
            selected_threshold = None
        st.markdown('___')

        ## Display Plots
        col1, col2 = st.columns((0.5, 0.5))
        col1.markdown("<h2 style='text-align: left; color: black;'>Silhouette Coefficient</h2>", unsafe_allow_html=True)
        col2.markdown("<h2 style='text-align: left; color: black;'>Cluster Visualization</h2>", unsafe_allow_html=True)

        if clustering_method == 'kmeans':
            # SH Score
            sh_score_plot = plot_sh_score(use_this_function=True, selected_cohort=selected_cohort,
                                          cohort_title=cohort_title, use_case_name='frontend', features_df=FEATURES_DF,
                                          selected_features=selected_features,
                                          selected_dependent_variable=selected_variable,
                                          use_encoding=True, clustering_method='kmeans', selected_cluster_count=selected_cluster_count, save_to_file=False)

            col1.pyplot(sh_score_plot, use_container_width=True)
            plt.clf()
            # Clustering
            clustering_plot = plot_k_means_on_pacmap(use_this_function=True,
                                                 display_sh_score=False,
                                                 selected_cohort=selected_cohort,
                                                 cohort_title=cohort_title,
                                                 use_case_name='frontend',
                                                 features_df=FEATURES_DF,
                                                 selected_features=selected_features,
                                                 selected_dependent_variable=selected_variable,
                                                 selected_cluster_count=selected_cluster_count,
                                                 use_encoding=True,
                                                 save_to_file=False)
            col2.pyplot(clustering_plot, use_container_width=True)
            plt.clf()

        elif clustering_method == 'kprototype':
            col1.write(f'Calculating the silhouette scores of kprototype for the first time can take up to 1 minute.')
            # SH Score
            sh_score_plot = plot_sh_score(use_this_function=True, selected_cohort=selected_cohort,
                                          cohort_title=cohort_title, use_case_name='frontend', features_df=FEATURES_DF,
                                          selected_features=selected_features,
                                          selected_dependent_variable=selected_variable,
                                          use_encoding=False,        # not needed for kprot
                                          clustering_method='kprot', selected_cluster_count=selected_cluster_count,
                                          save_to_file=False)

            col1.pyplot(sh_score_plot, use_container_width=True)
            plt.clf()

            # Clustering
            clustering_plot = plot_k_prot_on_pacmap(use_this_function=True,
                                                 display_sh_score=False,
                                                 selected_cohort=selected_cohort,
                                                 cohort_title=cohort_title,
                                                 use_case_name='frontend',
                                                 features_df=FEATURES_DF,
                                                 selected_features=selected_features,
                                                 selected_dependent_variable=selected_variable,
                                                 selected_cluster_count=selected_cluster_count,
                                                 use_encoding=False,        # not needed for kprot
                                                 save_to_file=False)
            col2.pyplot(clustering_plot, use_container_width=True)
            plt.clf()

        elif clustering_method == 'DBSCAN':
            col1.write(f'Iteratively change the DBSCAN parameters of epsilon and min_samples to optimize the clustering.')
            # SH Score
            sh_score_plot = plot_sh_score_DBSCAN(use_this_function=True, selected_cohort=selected_cohort,
                                          cohort_title=cohort_title, use_case_name='frontend', features_df=FEATURES_DF,
                                          selected_features=selected_features,
                                          selected_dependent_variable=selected_variable,
                                          selected_eps=selected_eps,
                                          selected_min_sample=selected_min_sample,
                                          use_encoding=True,
                                          save_to_file=False)

            col1.pyplot(sh_score_plot, use_container_width=True)
            plt.clf()

            # Clustering
            clustering_plot, dbscan_list = plot_DBSCAN_on_pacmap(use_this_function=True,
                                                display_sh_score=False,
                                                selected_cohort=selected_cohort,
                                                cohort_title=cohort_title,
                                                use_case_name='frontend',
                                                features_df=FEATURES_DF,
                                                selected_features=selected_features,
                                                selected_dependent_variable=selected_variable,
                                                selected_eps=selected_eps,
                                                selected_min_sample=selected_min_sample,
                                                use_encoding=True,  # not needed for kprot
                                                save_to_file=False)
            col2.pyplot(clustering_plot, use_container_width=True)
            plt.clf()

            if len(list(set(dbscan_list))) > 20:
                col2.warning('Warning: DBSCAN results in more than 20 clusters. Clustering might not be useful.')

        elif clustering_method == 'SLINK':
            # Get sh_score
            sh_score_plot = plot_sh_score_SLINK(use_this_function=True, selected_cohort=selected_cohort,
                                          cohort_title=cohort_title, use_case_name='frontend', features_df=FEATURES_DF,
                                          selected_features=selected_features,
                                          selected_dependent_variable=selected_variable,
                                          separation_criterion=selected_criterion,
                                          threshold=selected_threshold,
                                          use_encoding=True,
                                          save_to_file=False)
            col1.pyplot(sh_score_plot, use_container_width=True)
            plt.clf()

            # Get SLINK Clusters
            selected_cohort_preprocessed, clusters_list, slink_z, sh_score, pacmap_data_points = plot_SLINK_on_pacmap(use_this_function=True,
                                                                display_sh_score=False,
                                                                selected_cohort=selected_cohort,
                                                                cohort_title=cohort_title,
                                                                use_case_name='frontend',
                                                                features_df=FEATURES_DF,
                                                                selected_features=selected_features,
                                                                selected_dependent_variable=selected_variable,
                                                                use_encoding=True,
                                                                show_dendrogram=False,
                                                                separation_criterion=selected_criterion,
                                                                threshold=selected_threshold,
                                                                save_to_file=False)
            # Plot Clustering
            cluster_count = len(set(clusters_list))
            plot_title = f'SLINK_{cohort_title}_{selected_criterion}_t_{selected_threshold}'
            clustering_plot = plot_clusters_on_3D_pacmap(plot_title=plot_title, use_case_name='frontend',
                                            pacmap_data_points=pacmap_data_points, cluster_count=cluster_count,
                                            sh_score=sh_score, coloring=clusters_list, save_to_file=False)
            col2.pyplot(clustering_plot, use_container_width=True)
            plt.clf()

            if len(list(set(clusters_list))) > 20:
                col2.warning('Warning: SLINK results in more than 20 clusters. Clustering might not be useful.')

            # Additional: SLINK Dendrogram
            col2.markdown("<h2 style='text-align: left; color: black;'>SLINK Dendrogram</h2>", unsafe_allow_html=True)
            dendrogram_dict = dendrogram(Z=slink_z)
            Z = slink_z
            Z = np.asarray(Z, order='c')
            Zs = Z.shape
            n = Zs[0] + 1
            mh = max(Z[:, 2])
            dendrogram_fig, ax = plt.subplots()
            my_plot_dendrogram(icoords=dendrogram_dict['icoord'],
                                dcoords=dendrogram_dict['dcoord'],
                                ivl=dendrogram_dict['ivl'],
                                p=0,        # optional parameter for truncate mode
                                n=n,
                                mh=mh,
                                orientation='right',
                                no_labels=True,
                                color_list=dendrogram_dict['color_list'],
                                leaf_font_size=None,
                                leaf_rotation=None,
                                contraction_marks=None,
                                ax=ax,
                                above_threshold_color='C0')
            col2.pyplot(dendrogram_fig, use_container_width=True)
            plt.clf()

        else:
            st.warning('Please select a clustering option.')

    st.markdown('___')
