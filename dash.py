import pandas as pd
import streamlit as st
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix, recall_score
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
import shap
import json
import numpy as np
import plotly.express as px
import statsmodels.api as sm
import xgboost as xgb
def main():
    st.set_page_config(layout="wide", page_title="Анализ легендарности покемонов")
    data = pd.read_csv("data1.csv")

    def load_model():
        model = joblib.load('pokemons.pkl')
        return model

    def load_metrics():
        metrics_df = pd.read_csv('metrics.csv')
        return metrics_df
    metrics = load_metrics()
    model = load_model()

    def load_test_x():
        with open('test_data.json', 'r') as f:
            test_data = json.load(f)
        X_test = pd.DataFrame(test_data["x_test"])
        return X_test
    
    def load_test_y():
        with open('test_data.json', 'r') as f:
            test_data = json.load(f)
        y_test = pd.DataFrame(test_data["y_test"])
        return y_test
    
    def load_pred_y():
        with open('test_data.json', 'r') as f:
            test_data = json.load(f)
        y_pred = pd.DataFrame(test_data["y_pred"])
        return y_pred
    
    def load_features():
        with open('model_metadata.json', 'r') as f:
            test_data = json.load(f)
        return test_data["features"] 

    features = load_features()
    X_test = load_test_x()
    y_test = load_test_y()
    y_pred = load_pred_y()

    col1,col2,col3 = st.columns([7,7,7])

    with col2:
        st.title("Покемоны")
    st.title("Сташевич Герман Александрович 2023-ФГиИБ-ПИ-1б 19 Вариант")
    st.write("Цель моей работы: на основе характеристик покемонов анализировать их легендарность.")

    tab1, tab2, tab3, tab4 = st.tabs(["Исходные данные", "Графики зависимостей", "Матрица ошибок и метрики модели",  "Интерпретация результатов обучения модели"])

    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            st.subheader("SHAP-значения признаков")
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
            st.pyplot(fig, bbox_inches='tight')
            plt.close()
            #st.write('На графике видно что-то')
        
        with col2:
            st.subheader("SHAP-анализ")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            plt.figure(figsize=(6, 4))
            shap.summary_plot(
                shap_values, 
                X_test, 
                feature_names=features,
                plot_type="dot",
                show=False,
                max_display=min(20, len(features)))
            plt.tight_layout()
            st.pyplot(plt.gcf())
            plt.close()
            st.write('Интерпретация:')
            st.write('- Красные точки: высокие значения признака увеличивают вероятность класса "legend"')
            st.write('- Синие точки: низкие значения уменьшают вероятность класса "legend"')

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.header("1. Распределение признаков")
            feature = st.selectbox('Выберите признак:', [
                'against_bug', 'against_dark', 'against_dragon', 'against_electric', 'against_fairy', 'against_fight', 'against_fire', 'against_flying', 'against_ghost', 'against_grass', 'against_ground', 'against_ice', 'against_normal', 'against_poison', 'against_psychic', 'against_rock', 'against_steel', 'against_water',
    'attack', 'base_egg_steps', 'base_happiness', 'base_total', 'capture_rate',
       'defense', 'experience_growth', 'height_m', 'hp',
       'percentage_male', 'pokedex_number', 'sp_attack', 'sp_defense', 'speed', 'weight_kg',
       'type1_group1', 'type1_group10', 'type1_group2', 'type1_group3',
       'type1_group4', 'type1_group5', 'type1_group6', 'type1_group7',
       'type1_group8', 'type1_group9', 'type2_group1', 'type2_group2',
       'type2_group3', 'type2_group4', 'type2_group5', 'type2_group6',
       'type2_group7', 'generation_1', 'generation_2', 'generation_4',
       'generation_7', 'generation_3, 5, 6', 'Adaptability', 'Aftermath', 'Air Lock', 'Analytic', 'Anger Point', 'Anticipation', 'Arena Trap', 'Aroma Veil', 'Aura Break', 'Bad Dreams', 'Battery', 'Battle Armor', 'Battle Bond', 'Beast Boost', 'Berserk', 'Big Pecks', 'Blaze', 'Bulletproof', 'Cheek Pouch', 'Chlorophyll', 'Clear Body', 'Cloud Nine', 'Color Change', 'Comatose', 'Competitive', 'Compoundeyes', 'Contrary', 'Corrosion', 'Cursed Body', 'Cute Charm', 'Damp', 'Dancer', 'Dark Aura', 'Dazzling', 'Defeatist', 'Defiant', 'Disguise', 'Download', 'Drizzle', 'Drought', 'Dry Skin', 'Early Bird', 'Effect Spore', 'Electric Surge', 'Emergency Exit', 'Fairy Aura', 'Filter', 'Flame Body', 'Flare Boost', 'Flash Fire', 'Flower Gift', 'Flower Veil', 'Fluffy', 'Forecast', 'Forewarn', 'Friend Guard', 'Frisk', 'Full Metal Body', 'Fur Coat', 'Gale Wings', 'Galvanize', 'Gluttony', 'Gooey', 'Grass Pelt', 'Grassy Surge', 'Guts', 'Harvest', 'Healer', 'Heatproof', 'Heavy Metal', 'Honey Gather', 'Huge Power', 'Hustle', 'Hydration', 'Hyper Cutter', 'Ice Body', 'Illuminate', 'Illusion', 'Immunity', 'Imposter', 'Infiltrator', 'Innards Out', 'Inner Focus', 'Insomnia', 'Intimidate', 'Iron Barbs', 'Iron Fist', 'Justified', 'Keen Eye', 'Klutz', 'Leaf Guard', 'Levitate', 'Light Metal', 'Lightningrod', 'Limber', 'Liquid Ooze', 'Liquid Voice', 'Long Reach', 'Magic Bounce', 'Magic Guard', 'Magician', 'Magma Armor', 'Magnet Pull', 'Marvel Scale', 'Mega Launcher', 'Merciless', 'Minus', 'Misty Surge', 'Mold Breaker', 'Moody', 'Motor Drive', 'Moxie', 'Multiscale', 'Multitype', 'Mummy', 'Natural Cure', 'No Guard', 'Normalize', 'Oblivious', 'Overcoat', 'Overgrow', 'Own Tempo', 'Pickpocket', 'Pickup', 'Pixilate', 'Plus', 'Poison Heal', 'Poison Point', 'Poison Touch', 'Power Construct', 'Power of Alchemy', 'Prankster', 'Pressure', 'Prism Armor', 'Protean', 'Psychic Surge', 'Pure Power', 'Queenly Majesty', 'Quick Feet', 'RKS System', 'Rain Dish', 'Rattled', 'Receiver', 'Reckless', 'Refrigerate', 'Regenerator', 'Rivalry', 'Rock Head', 'Rough Skin', 'Run Away', 'Sand Force', 'Sand Rush', 'Sand Stream', 'Sand Veil', 'Sap Sipper', 'Schooling', 'Scrappy', 'Serene Grace', 'Shadow Shield', 'Shadow Tag', 'Shed Skin', 'Sheer Force', 'Shell Armor', 'Shield Dust', 'Shields Down', 'Simple', 'Skill Link', 'Slow Start', 'Slush Rush', 'Sniper', 'Snow Cloak', 'Snow Warning', 'Solar Power', 'Solid Rock', 'Soul-Heart', 'Soundproof', 'Speed Boost', 'Stakeout', 'Stall', 'Stamina', 'Stance Change', 'Static', 'Steadfast', 'Steelworker', 'Stench', 'Sticky Hold', 'Storm Drain', 'Strong Jaw', 'Sturdy', 'Suction Cups', 'Super Luck', 'Surge Surfer', 'Swarm', 'Sweet Veil', 'Swift Swim', 'Symbiosis', 'Synchronize', 'Tangled Feet', 'Tangling Hair', 'Technician', 'Telepathy', 'Teravolt', 'Thick Fat', 'Tinted Lens', 'Torrent', 'Tough Claws', 'Toxic Boost', 'Trace', 'Triage', 'Truant', 'Turboblaze', 'Unaware', 'Unburden', 'Unnerve', 'Victory Star', 'Vital Spirit', 'Volt Absorb', 'Water Absorb', 'Water Bubble', 'Water Compaction', 'Water Veil', 'Weak Armor', 'White Smoke', 'Wimp Out', 'Wonder Guard', 'Wonder Skin ', 'Zen Mode'
            ])
            color_scale = alt.Scale(
                domain=['legend', 'no_legend'],
                range=['green','red']
            )
            sorted_data = data.sort_values('is_legendary', ascending=[False])
            hist = alt.Chart(sorted_data).mark_bar(
                opacity=0.5, 
                binSpacing=0, stroke='black',
            strokeWidth=0.5
            ).encode(
                alt.X(f'{feature}:Q').bin(maxbins=50),
                alt.Y('count()').stack(None),
                alt.Color('is_legendary:N', scale=color_scale, 
                        legend=alt.Legend(title="Легендарность")),
                tooltip=['count()', 'is_legendary'],
                order=alt.Order('is_legendary', sort='descending')
            ).properties(
                width=600,
                height=400
            ).interactive()
            st.altair_chart(hist, use_container_width=True)

        with col2:
            st.header("2. Зависимость")
            x_axis = st.selectbox(
            'Выберите признак для оси X:',
            [
                'against_bug', 'against_dark', 'against_dragon', 'against_electric', 'against_fairy', 'against_fight', 'against_fire', 'against_flying', 'against_ghost', 'against_grass', 'against_ground', 'against_ice', 'against_normal', 'against_poison', 'against_psychic', 'against_rock', 'against_steel', 'against_water',
    'attack', 'base_egg_steps', 'base_happiness', 'base_total', 'capture_rate',
       'defense', 'experience_growth', 'height_m', 'hp',
       'percentage_male', 'pokedex_number', 'sp_attack', 'sp_defense', 'speed', 'weight_kg',
       'type1_group1', 'type1_group10', 'type1_group2', 'type1_group3',
       'type1_group4', 'type1_group5', 'type1_group6', 'type1_group7',
       'type1_group8', 'type1_group9', 'type2_group1', 'type2_group2',
       'type2_group3', 'type2_group4', 'type2_group5', 'type2_group6',
       'type2_group7', 'generation_1', 'generation_2', 'generation_4',
       'generation_7', 'generation_3, 5, 6', 'Adaptability', 'Aftermath', 'Air Lock', 'Analytic', 'Anger Point', 'Anticipation', 'Arena Trap', 'Aroma Veil', 'Aura Break', 'Bad Dreams', 'Battery', 'Battle Armor', 'Battle Bond', 'Beast Boost', 'Berserk', 'Big Pecks', 'Blaze', 'Bulletproof', 'Cheek Pouch', 'Chlorophyll', 'Clear Body', 'Cloud Nine', 'Color Change', 'Comatose', 'Competitive', 'Compoundeyes', 'Contrary', 'Corrosion', 'Cursed Body', 'Cute Charm', 'Damp', 'Dancer', 'Dark Aura', 'Dazzling', 'Defeatist', 'Defiant', 'Disguise', 'Download', 'Drizzle', 'Drought', 'Dry Skin', 'Early Bird', 'Effect Spore', 'Electric Surge', 'Emergency Exit', 'Fairy Aura', 'Filter', 'Flame Body', 'Flare Boost', 'Flash Fire', 'Flower Gift', 'Flower Veil', 'Fluffy', 'Forecast', 'Forewarn', 'Friend Guard', 'Frisk', 'Full Metal Body', 'Fur Coat', 'Gale Wings', 'Galvanize', 'Gluttony', 'Gooey', 'Grass Pelt', 'Grassy Surge', 'Guts', 'Harvest', 'Healer', 'Heatproof', 'Heavy Metal', 'Honey Gather', 'Huge Power', 'Hustle', 'Hydration', 'Hyper Cutter', 'Ice Body', 'Illuminate', 'Illusion', 'Immunity', 'Imposter', 'Infiltrator', 'Innards Out', 'Inner Focus', 'Insomnia', 'Intimidate', 'Iron Barbs', 'Iron Fist', 'Justified', 'Keen Eye', 'Klutz', 'Leaf Guard', 'Levitate', 'Light Metal', 'Lightningrod', 'Limber', 'Liquid Ooze', 'Liquid Voice', 'Long Reach', 'Magic Bounce', 'Magic Guard', 'Magician', 'Magma Armor', 'Magnet Pull', 'Marvel Scale', 'Mega Launcher', 'Merciless', 'Minus', 'Misty Surge', 'Mold Breaker', 'Moody', 'Motor Drive', 'Moxie', 'Multiscale', 'Multitype', 'Mummy', 'Natural Cure', 'No Guard', 'Normalize', 'Oblivious', 'Overcoat', 'Overgrow', 'Own Tempo', 'Pickpocket', 'Pickup', 'Pixilate', 'Plus', 'Poison Heal', 'Poison Point', 'Poison Touch', 'Power Construct', 'Power of Alchemy', 'Prankster', 'Pressure', 'Prism Armor', 'Protean', 'Psychic Surge', 'Pure Power', 'Queenly Majesty', 'Quick Feet', 'RKS System', 'Rain Dish', 'Rattled', 'Receiver', 'Reckless', 'Refrigerate', 'Regenerator', 'Rivalry', 'Rock Head', 'Rough Skin', 'Run Away', 'Sand Force', 'Sand Rush', 'Sand Stream', 'Sand Veil', 'Sap Sipper', 'Schooling', 'Scrappy', 'Serene Grace', 'Shadow Shield', 'Shadow Tag', 'Shed Skin', 'Sheer Force', 'Shell Armor', 'Shield Dust', 'Shields Down', 'Simple', 'Skill Link', 'Slow Start', 'Slush Rush', 'Sniper', 'Snow Cloak', 'Snow Warning', 'Solar Power', 'Solid Rock', 'Soul-Heart', 'Soundproof', 'Speed Boost', 'Stakeout', 'Stall', 'Stamina', 'Stance Change', 'Static', 'Steadfast', 'Steelworker', 'Stench', 'Sticky Hold', 'Storm Drain', 'Strong Jaw', 'Sturdy', 'Suction Cups', 'Super Luck', 'Surge Surfer', 'Swarm', 'Sweet Veil', 'Swift Swim', 'Symbiosis', 'Synchronize', 'Tangled Feet', 'Tangling Hair', 'Technician', 'Telepathy', 'Teravolt', 'Thick Fat', 'Tinted Lens', 'Torrent', 'Tough Claws', 'Toxic Boost', 'Trace', 'Triage', 'Truant', 'Turboblaze', 'Unaware', 'Unburden', 'Unnerve', 'Victory Star', 'Vital Spirit', 'Volt Absorb', 'Water Absorb', 'Water Bubble', 'Water Compaction', 'Water Veil', 'Weak Armor', 'White Smoke', 'Wimp Out', 'Wonder Guard', 'Wonder Skin ', 'Zen Mode'
            ],
             index=5,
            key='x_axis'
        )
            y_axis = st.selectbox(
                    'Выберите признак для оси Y:',
                    [
                        'against_bug', 'against_dark', 'against_dragon', 'against_electric', 'against_fairy', 'against_fight', 'against_fire', 'against_flying', 'against_ghost', 'against_grass', 'against_ground', 'against_ice', 'against_normal', 'against_poison', 'against_psychic', 'against_rock', 'against_steel', 'against_water',
    'attack', 'base_egg_steps', 'base_happiness', 'base_total', 'capture_rate',
       'defense', 'experience_growth', 'height_m', 'hp',
       'percentage_male', 'pokedex_number', 'sp_attack', 'sp_defense', 'speed', 'weight_kg',
       'type1_group1', 'type1_group10', 'type1_group2', 'type1_group3',
       'type1_group4', 'type1_group5', 'type1_group6', 'type1_group7',
       'type1_group8', 'type1_group9', 'type2_group1', 'type2_group2',
       'type2_group3', 'type2_group4', 'type2_group5', 'type2_group6',
       'type2_group7', 'generation_1', 'generation_2', 'generation_4',
       'generation_7', 'generation_3, 5, 6', 'Adaptability', 'Aftermath', 'Air Lock', 'Analytic', 'Anger Point', 'Anticipation', 'Arena Trap', 'Aroma Veil', 'Aura Break', 'Bad Dreams', 'Battery', 'Battle Armor', 'Battle Bond', 'Beast Boost', 'Berserk', 'Big Pecks', 'Blaze', 'Bulletproof', 'Cheek Pouch', 'Chlorophyll', 'Clear Body', 'Cloud Nine', 'Color Change', 'Comatose', 'Competitive', 'Compoundeyes', 'Contrary', 'Corrosion', 'Cursed Body', 'Cute Charm', 'Damp', 'Dancer', 'Dark Aura', 'Dazzling', 'Defeatist', 'Defiant', 'Disguise', 'Download', 'Drizzle', 'Drought', 'Dry Skin', 'Early Bird', 'Effect Spore', 'Electric Surge', 'Emergency Exit', 'Fairy Aura', 'Filter', 'Flame Body', 'Flare Boost', 'Flash Fire', 'Flower Gift', 'Flower Veil', 'Fluffy', 'Forecast', 'Forewarn', 'Friend Guard', 'Frisk', 'Full Metal Body', 'Fur Coat', 'Gale Wings', 'Galvanize', 'Gluttony', 'Gooey', 'Grass Pelt', 'Grassy Surge', 'Guts', 'Harvest', 'Healer', 'Heatproof', 'Heavy Metal', 'Honey Gather', 'Huge Power', 'Hustle', 'Hydration', 'Hyper Cutter', 'Ice Body', 'Illuminate', 'Illusion', 'Immunity', 'Imposter', 'Infiltrator', 'Innards Out', 'Inner Focus', 'Insomnia', 'Intimidate', 'Iron Barbs', 'Iron Fist', 'Justified', 'Keen Eye', 'Klutz', 'Leaf Guard', 'Levitate', 'Light Metal', 'Lightningrod', 'Limber', 'Liquid Ooze', 'Liquid Voice', 'Long Reach', 'Magic Bounce', 'Magic Guard', 'Magician', 'Magma Armor', 'Magnet Pull', 'Marvel Scale', 'Mega Launcher', 'Merciless', 'Minus', 'Misty Surge', 'Mold Breaker', 'Moody', 'Motor Drive', 'Moxie', 'Multiscale', 'Multitype', 'Mummy', 'Natural Cure', 'No Guard', 'Normalize', 'Oblivious', 'Overcoat', 'Overgrow', 'Own Tempo', 'Pickpocket', 'Pickup', 'Pixilate', 'Plus', 'Poison Heal', 'Poison Point', 'Poison Touch', 'Power Construct', 'Power of Alchemy', 'Prankster', 'Pressure', 'Prism Armor', 'Protean', 'Psychic Surge', 'Pure Power', 'Queenly Majesty', 'Quick Feet', 'RKS System', 'Rain Dish', 'Rattled', 'Receiver', 'Reckless', 'Refrigerate', 'Regenerator', 'Rivalry', 'Rock Head', 'Rough Skin', 'Run Away', 'Sand Force', 'Sand Rush', 'Sand Stream', 'Sand Veil', 'Sap Sipper', 'Schooling', 'Scrappy', 'Serene Grace', 'Shadow Shield', 'Shadow Tag', 'Shed Skin', 'Sheer Force', 'Shell Armor', 'Shield Dust', 'Shields Down', 'Simple', 'Skill Link', 'Slow Start', 'Slush Rush', 'Sniper', 'Snow Cloak', 'Snow Warning', 'Solar Power', 'Solid Rock', 'Soul-Heart', 'Soundproof', 'Speed Boost', 'Stakeout', 'Stall', 'Stamina', 'Stance Change', 'Static', 'Steadfast', 'Steelworker', 'Stench', 'Sticky Hold', 'Storm Drain', 'Strong Jaw', 'Sturdy', 'Suction Cups', 'Super Luck', 'Surge Surfer', 'Swarm', 'Sweet Veil', 'Swift Swim', 'Symbiosis', 'Synchronize', 'Tangled Feet', 'Tangling Hair', 'Technician', 'Telepathy', 'Teravolt', 'Thick Fat', 'Tinted Lens', 'Torrent', 'Tough Claws', 'Toxic Boost', 'Trace', 'Triage', 'Truant', 'Turboblaze', 'Unaware', 'Unburden', 'Unnerve', 'Victory Star', 'Vital Spirit', 'Volt Absorb', 'Water Absorb', 'Water Bubble', 'Water Compaction', 'Water Veil', 'Weak Armor', 'White Smoke', 'Wimp Out', 'Wonder Guard', 'Wonder Skin ', 'Zen Mode'
                    ],
                    index=4, 
                    key='y_axis'
                )
            
            st.write(f"**Зависимость {y_axis} от {x_axis}**")
            
            fig = px.scatter(
                data,
                x=x_axis,
                y=y_axis,
                color='is_legendary',
                color_discrete_map={'legend': 'green', 'no_legend': 'red'},
                trendline="lowess",
                trendline_options=dict(frac=0.3),
                width=800,
                height=500
            )
            
            fig.update_layout(
                xaxis_title=f"{x_axis}",
                yaxis_title=f"{y_axis}",
                legend_title="Легендарность",
                hovermode='closest'
            )
            
            fig.update_traces(
                line=dict(width=4),
                marker=dict(size=1, opacity=0.5)
            )
            st.plotly_chart(fig, use_container_width=True)


    with tab3:
        final_matrix = confusion_matrix(y_test, y_pred)

        fig = px.imshow(final_matrix,
                   labels=dict(x="Предсказано", y="Истинное", color="Count"),
                   x=['Нелегендарный', 'Легендарный'],
                   y=['Нелегендарный', 'Легендарный'],
                   text_auto=True,
                   color_continuous_scale='Greens')
        fig.update_layout(title='Матрица ошибок')
        st.plotly_chart(fig)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Тестовая Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
        
        with col2:
            st.metric("Test F1-score", f"{f1_score(y_test, y_pred):.4f}")
        
        with col3:
            st.metric("recall", f"{recall_score(y_test, y_pred):.4f}")
        #st.subheader("Отчёт классификации")
        #report = classification_report(y_test, y_pred, output_dict=True)
        #st.table(pd.DataFrame(report).transpose())



    with tab1:
        st.dataframe(data)

    return data


if __name__ == '__main__':
    main()


