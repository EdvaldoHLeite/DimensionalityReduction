import pandas as pd
import numpy as np
from scipy.io import arff
########################################### Duas classes ##########################################



#############################################3
###################     ERROR       #################
# header deve ser None, se nao ele pega a primeira linha e diz ser o cabecalho
##########################3


def pima():
    df = pd.read_csv('bases/pima/pima-indians-diabetes.csv', sep=',')
    df.columns = ['a'+str(x) for x in range(8)] + ['Class']
    
    # possui os dados
    x = np.array(df.drop('Class',1))

    # possui as classes
    y = np.array(df.Class)

    return [x, y, 'Pima', df.drop('Class',1).columns]

def banknote():
    df = pd.read_csv('bases/banknote/data_banknote_authentication.txt')

    df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'target']

    # possui os dados
    x = np.array(df.drop('target',1))

    # possui as classes
    y = np.array(df.target)

    return [x, y, 'Banknote', df.drop('target',1).columns]
    
def climate():        
    df = pd.read_table("bases/climate/pop_failures.dat", sep="\s+")  # usa um ou mais espacos
    #np.nan_to_num(df)

    df = df.drop('Study', 1) # as ids serao apagadas
    df = df.drop('Run', 1)
    
    x = np.array(df.drop('outcome',1)) # axis=0 row e axis=1 columns    
    # possui as classes
    y = np.array(df.outcome)

    # 18 features
    return [x, y, 'Climate', df.drop('outcome',1).columns]
    
def debrecen():
    dados = arff.loadarff('bases/debrecen/messidor_features.arff')
    df = pd.DataFrame(dados[0])
    
    x = np.array(df.drop('Class',1)) # axis=0 row e axis=1 columns    
    # possui as classes
    y = np.array(df.Class)
    y=y.astype('int')
    return [x, y, 'Debrecen', df.drop('Class',1).columns]
    
def occupancy():
    df = pd.read_csv('bases/occupancy/datatraining.txt')

    df = df.drop('date', 1) # apaga o 'date'

    # possui os dados
    x = np.array(df.drop('Occupancy',1))
    # possui as classes
    y = np.array(df.Occupancy)
    # 5 features
    return [x, y, 'Occupancy', df.drop('Occupancy',1).columns]
    
def vcolumn():
    df = pd.read_csv('bases/vcolumn/column_2C.dat', sep='\s+')
    # adiciona o cabecalho
    df.columns = ['dado1', 'dado2', 'dado3', 'dado4', 'dado5', 'dado6', 'Class']
    #df['date'] = df['date'].replace(lambda x: (data_to_number(x)))
    
    # possui os dados
    df['Class'] = df['Class'].replace(['AB', 'NO'], [1, 0])
    x = np.array(df.drop('Class',1))

    # possui as classes
    y = np.array(df.Class)
    # 6 features
    return [x, y, 'VColumn', df.drop('Class',1).columns]
    
    
def spambase():
    df = pd.read_csv('bases/spambase/spambase.data', sep=',', header=None)
        # adiciona o cabecalho
    df.columns = ['word_freq_make', 
            'word_freq_address', 
            'word_freq_all', 
            'word_freq_3d', 
            'word_freq_our', 
            'word_freq_over', 
            'word_freq_remove', 
            'word_freq_internet', 
            'word_freq_order', 
            'word_freq_mail', 
            'word_freq_receive', 
            'word_freq_will', 
            'word_freq_people', 
            'word_freq_report', 
            'word_freq_addresses', 
            'word_freq_free', 
            'word_freq_business', 
            'word_freq_email', 
            'word_freq_you', 
            'word_freq_credit', 
            'word_freq_your', 
            'word_freq_font', 
            'word_freq_000', 
            'word_freq_money', 
            'word_freq_hp', 
            'word_freq_hpl', 
            'word_freq_george', 
            'word_freq_650', 
            'word_freq_lab', 
            'word_freq_labs', 
            'word_freq_telnet', 
            'word_freq_857', 
            'word_freq_data', 
            'word_freq_415', 
            'word_freq_85', 
            'word_freq_technology', 
            'word_freq_1999', 
            'word_freq_parts', 
            'word_freq_pm', 
            'word_freq_direct', 
            'word_freq_cs', 
            'word_freq_meeting', 
            'word_freq_original', 
            'word_freq_project', 
            'word_freq_re', 
            'word_freq_edu', 
            'word_freq_table', 
            'word_freq_conference', 
            'char_freq_;', 
            'char_freq_(', 
            'char_freq_[', 
            'char_freq_!', 
            'char_freq_$', 
            'char_freq_#', 
            'capital_run_length_average', 
            'capital_run_length_longest', 
            'capital_run_length_total', 
            'Class']

        # possui os dados
    x = np.array(df.drop('Class',1))

        # possui as classes
    y = np.array(df.Class)
    # 6 features
    return [x, y, 'Spambase', df.drop('Class',1).columns]
    
def wdbc():
    df = pd.read_csv('bases/wdbc/wdbc.data', sep=',', header=None)
    # adiciona o cabecalho
    df.columns = ['ID', 'Diagnosis', 'dado1', 'dado2', 'dado3', 'dado4', 'dado5', 'dado6', 'dado7', 'dado8', 'dado9', 'dado10', 'dado11', 'dado12', 'dado13', 'dado14', 'dado15', 'dado16', 'dado17', 'dado18', 'dado19', 'dado20', 'dado21', 'dado22', 'dado23', 'dado24', 'dado25', 'dado26', 'dado27', 'dado28', 'dado29', 'dado30']
    #df['date'] = df['date'].replace(lambda x: (data_to_number(x)))
    
    df['Diagnosis'] = df['Diagnosis'].replace(['M', 'B'], [1, 0])
    
    # possui os dados
    x = np.array((df.drop('ID',1)).drop('Diagnosis', 1)) # o id nao interessa o diagnostico eh a classe

    # possui as classes
    y = np.array(df.Diagnosis)
    # 6 features
    return [x, y, 'WDBC', (df.drop('ID',1)).drop('Diagnosis', 1).columns]
    
#### bases do parzen

def survival():
    df = pd.read_csv('bases/survival/haberman.data')
    df.columns = ['age', 'year', 'nodes', 'Class']
    
    x = np.array(df.drop('Class', 1))
    y = np.array(df.Class)
    
    return [x, y, 'Survival', df.drop('Class', 1).columns]    

def monks():
    # monks-1.test possui os exemplos completos
    df = pd.read_csv('bases/monks/monks-1.test', sep='\s+') # usa o primeiro teste
    df.columns = ["Class", 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'id']
        
    df = df.drop('id', 1) # id eh inutil entao eh removida
    
    x = np.array(df.drop('Class', 1))
    y = np.array(df.Class)
    
    return [x, y, 'Monks', df.drop('Class', 1).columns]

def immunotherapy():
    df = pd.read_csv('bases/Immunotherapy/Immunotherapy.txt', sep='\s+') # usa o primeiro teste
    
    x = np.array(df.drop('Result_of_Treatment', 1))
    y = np.array(df.Result_of_Treatment)
    
    return [x, y, 'Immunotherapy', df.drop('Result_of_Treatment', 1).columns]

def titanic():
    df = pd.read_csv('bases/Titanic/train_clean.csv', sep=',')
    ### binarizacao de embarked
    df['Q'] = [int(x=='Q') for x in df['Embarked']]
    df['S'] = [int(x=='S') for x in df['Embarked']]
    df['C'] = [int(x=='C') for x in df['Embarked']]

    ## removendo atributos
    df = df.drop('Embarked', 1)
    df = df.drop('Cabin', 1)
    df = df.drop('Name', 1)
    df = df.drop("PassengerId", 1)
    df = df.drop("Ticket", 1)
    df = df.drop("Title", 1)

    ## binarizando sex
    df['Sex'] = df['Sex'].replace(['male', 'female'], [1, 0])

    x = np.array(df.drop('Survived', 1))
    y = np.array(df.Survived)

    return [x, y, 'Titanic', df.drop('Survived', 1).columns]

def wine():
    df = pd.read_csv('bases/Wine/wine.data')
    
    df.columns = ['Class','Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium',',Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline']
    
    x = np.array(df.drop("Class", 1))
    y = np.array(df.Class)
    
    return [x, y, 'Wine', df.drop("Class", 1).columns]
    
def hill_valley():
    df_train = pd.read_csv('bases/HillValley/Hill_Valley_without_noise_Training.data')
    df_test = pd.read_csv('bases/HillValley/Hill_Valley_without_noise_Testing.data')
    
    x_train = np.array(df_train.drop("Class", 1))
    x_test = np.array(df_test.drop('Class', 1))
    y_train = np.array(df_train.Class)
    y_test = np.array(df_test.Class)

    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    
    return [x, y, 'HillValley', df_train.drop("Class", 1).columns]

######################################### Mais de duas classes #############################################

def dermatology():
    df = pd.read_csv('bases/Dermatology/dermatology.data')
    colunas = [str(i+1) for i in range(34)]
    colunas.append('Class')
    df.columns = colunas 
    
    df['34'] = df['34'].replace('?', -1)
    for y in df['34']:
        df['34'] = df['34'].replace(y, int(y))
    #print(type(df['34'][5]))
    x = np.array(df.drop('Class', 1))
    y = np.array(df.Class)
    
    return [x, y, 'Dermatology']

def leaf():
    df = pd.read_csv('bases/Leaf/leaf.csv')
    df.columns = ['Class','a1','a2','a3','a4','a5','a6','a7','a8','a9','a10','a11','a12','a13','a14','a15']
    
    x = np.array(df.drop('Class', 1))
    y = np.array(df.Class)
    
    return [x, y, 'Leaf']


def letter():
    df = pd.read_csv('bases/letter/letter-recognition.data', header=None)
    df.columns = ['lettr', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15', 'a16']
    letras = ['A', 'B','C', 'D','E','F','G','H','I','J','K','L','M','N','O','P','Q', 'R','S','T','U','V','W','X','Y','Z']
    df['lettr'] = df['lettr'].replace(letras, [x for x in range(len(letras))])

    x = np.array(df.drop("lettr", 1))
    y = np.array(df.lettr)
    
    return [x, y, 'Letter', df.drop('lettr', 1).columns]

def obs_network():
    dados = arff.loadarff('bases/obs_network/OBS-Network-DataSet_2_Aug27.arff')
    df = pd.DataFrame(dados[0])

    df['Class'] = df['Class'].replace([b'NB-No Block',b'Block',b'No Block',b'NB-Wait'], [0, 1, 2, 3])
    # comportamento na rede, b: comportando-se-1, nb: nao se comportando -- -1 e talvez nao se comportando pnb:0
    df['Node Status'] = df['Node Status'].replace([b'B', b'NB', b'P NB'], [1, -1, 0])
    
    # converte para int
    head = df.head()    
    for x in head:
        df[x] = df[x].astype(float)
    
    # preenche os NaNs com a media
    df = df.fillna(df.mean(0))
    
    x = np.array(df.drop('Class', 1))
    y = np.array(df.Class)
    return [x, y, 'obs_network', df.drop('Class', 1).columns]

def mice():
    df = pd.read_excel('bases/mice/Data_Cortex_Nuclear.xls')
    
    treatment = df['Treatment']
    genotype = df['Genotype']
    behavior = df['Behavior']
    Control = []
    Ts65Dn = []
    Memantine = []
    Saline = []
    CS = []
    SC = []
    for i in range(len(df)):        
        # genotype
        Control.append(int(genotype[i]  == 'Control'))
        Ts65Dn.append(int(genotype[i] == 'Ts65Dn'))
        
        # treatment
        Memantine.append(int(treatment[i] == 'Memantine'))
        Saline.append(int (treatment[i] == 'Saline'))
        
        # bahavior
        CS.append(int(behavior[i] == 'C/S'))
        SC.append(int(behavior[i] == 'S/C'))
    

    df['Control'] = Control
    df['Ts65Dn'] = Ts65Dn
    df['Memantine'] = Memantine
    df['Saline'] = Saline
    df['C/S'] = CS
    df['S/C'] = SC
    
    # apagando binarios
    df = df.drop(['Treatment', 'Genotype', 'Behavior'], 1)
    # apagando ids
    df = df.drop('MouseID', 1)

    '''#deletando linhas com valores falhos
    df = df.dropna(axis=0)'''

    # deletando colunas com valores falhos, para impedir que nao sejam eliminadas muitas amostras
    # mesmo que algumas colunas sejam eliminadas
    df = df.dropna(axis=1)
    
    # ajustando nomes das classes
    df.rename(columns={'class':"Class"}, inplace=True)
    df['Class'] = df['Class'].replace(['c-CS-m', 'c-SC-m', 'c-CS-s', 'c-SC-s', 
        't-CS-m','t-SC-m', 't-CS-s', 't-SC-s'], [0, 1, 2, 3, 4, 5, 6, 7])
    
    #print(df.head)
    head = df.head()    
    for x in head:
        df[x] = df[x].astype(float)
    
    x = np.array(df.drop('Class', 1))
    y = np.array(df.Class)
    
    return [x, y, 'Mice', df.drop('Class', 1).columns]

def user_knowledge():
    # uso de excel para import
    df_train = pd.read_excel('bases/UserKnowledge/Data_User_Modeling_Dataset_Hamdi Tolga KAHRAMAN.xls', sheet_name='Training_Data')
    df_test = pd.read_excel('bases/UserKnowledge/Data_User_Modeling_Dataset_Hamdi Tolga KAHRAMAN.xls', sheet_name='Test_Data')
    # concatenando o treino e teste das duas planilhas de um unico arquivo
    df = df_train.append(df_test, ignore_index=True)
    df.rename(columns={" UNS":"UNS"}, inplace=True)

    #Deletando colunas extras ou vazias
    df = df.dropna(axis=1)
    #df = df.drop('Attribute Information:', axis=1)

    df['UNS'] = df['UNS'].replace(['very_low','Low','Middle','High'], [0, 1, 2, 3])
    df['UNS'] = df['UNS'].replace(['Very Low'], [0]) # very low parece estar diferente

    x = np.array(df.drop("UNS", 1))
    y = np.array(df.UNS)
    
    return [x, y, 'UserKnowledge', df.drop("UNS", 1).columns]

def wine_quality():
    df = pd.read_csv('bases/winequality/winequality.csv', sep=';')

    df.columns = ['fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide','density','pH','sulphates','alcohol','quality']
    x = np.array(df.drop("quality", 1))
    y = np.array(df.quality)

    return [x, y, 'Wine Quality']
    
def wine_quality_red():
    df = pd.read_csv('bases/winequality/winequality-red.csv', sep=';')

    df.columns = ['fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide','density','pH','sulphates','alcohol','quality']
    x = np.array(df.drop("quality", 1))
    y = np.array(df.quality)

    return [x, y, 'winequality-red', df.drop("quality", 1).columns]

def wine_quality_white():
    df = pd.read_csv('bases/winequality/winequality-white.csv', sep=';')

    df.columns = ['fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide','density','pH','sulphates','alcohol','quality']
    x = np.array(df.drop("quality", 1))
    y = np.array(df.quality)

    return [x, y, 'winequality-white', df.drop("quality", 1).columns]

def lsvt():
    df = pd.read_excel('bases/LSVTVoiceRehabilitation/lsvt.xlsx', enconding='utf-8')
    df['Class'] = df['Class'].replace([1, 2], [0, 1])

    x = np.array(df.drop('Class', 1))
    y = np.array(df.Class)

    return [x, y, 'LSVT']

def spectf():
    df_test = pd.read_csv('bases/spectf/SPECTF.test', sep=',')
    df_train = pd.read_csv('bases/spectf/SPECTF.train', sep=',')
    
    df_test.columns = ['Class'] + [str(x) for x in range(44)] # criando cabecalho
    df_train.columns = ['Class'] + [str(x) for x in range(44)] # criando cabecalho
    
    teste_x = np.array(df_test.drop('Class', 1))
    treino_x = np.array(df_train.drop('Class', 1))
    teste_y = np.array(df_test.Class)
    treino_y = np.array(df_train.Class)
    
    
    x = np.concatenate((teste_x, treino_x), axis=0)
    y = np.concatenate((teste_y, treino_y), axis=0)

    return [x, y, 'SPECTF']

def bank():
    df = pd.read_csv('bases/bank/bank-full.csv', sep=';')

    # troca os atributos nao numericos

    # job
    # 12 novas features, menos 1
    df['job_admin'] = [int(x=='admin.') for x in df['job']]
    df['job_unknown'] = [int(x=='unknown') for x in df['job']]
    df['job_unemployed'] = [int(x=='unemployed') for x in df['job']]
    df['job_management'] = [int(x=='management') for x in df['job']]
    df['job_housemaid'] = [int(x=='housemaid') for x in df['job']]
    df['job_entrepreneur'] = [int(x=='entrepreneur') for x in df['job']]
    df['job_student'] = [int(x=='student') for x in df['job']]
    df['job_blue-collar'] = [int(x=='blue-collar') for x in df['job']]
    df['job_self-employed'] = [int(x=='self-employed') for x in df['job']]
    df['job_retired'] = [int(x=='retired') for x in df['job']]
    df['job_technician'] = [int(x=='technician') for x in df['job']]
    df['job_services'] = [int(x=='services') for x in df['job']]
    df = df.drop('job', 1)

    '''df['job_admin'] = []
    df['job_unknown'] = []
    df['job_unemployed'] = []
    df['job_management'] = []
    df['job_housemaid'] = []
    df['job_entrepreneur'] = []
    df['job_student'] = []
    df['job_blue-collar'] = []
    df['job_self-employed'] = []
    df['job_retired'] = []
    df['job_technician'] = []
    df['job_services'] = []
    for x in df['job']:
        df['job_admin'].append(int(x=='admin.'))
        df['job_unknown'].append(int(x=='unknown'))
        df['job_unemployed'].append(int(x=='unemployed'))
        df['job_management'].append(int(x=='management'))
        df['job_housemaid'].append(int(x=='housemaid'))
        df['job_entrepreneur'].append(int(x=='entrepreneur'))
        df['job_student'].append(int(x=='student'))
        df['job_blue-collar'].append(int(x=='blue-collar'))
        df['job_self-employed'].append(int(x=='self-employed'))
        df['job_retired'].append(int(x=='retired'))
        df['job_technician'].append(int(x=='technician'))
        df['job_services'].append(int(x=='services'))
    df.drop('job', 1)   ''' 

    # marital
    # 3 novas
    df['marital_married'] = [int(x=='married') for x in df['marital']]
    df['marital_divorced'] = [int(x=='divorced') for x in df['marital']]
    df['marital_single'] = [int(x=='single') for x in df['marital']]
    df = df.drop('marital', 1)

    '''df['marital_married'] = []
    df['marital_divorced'] = []
    df['marital_single'] = []
    for x in df['marital']:
        df['marital_married'].append(int(x=='married'))
        df['marital_divorced'].append(int(x=='divorced'))
        df['marital_single'].append(int(x=='single'))
    df = df.drop('marital', 1)'''
    
    # education
    # 4 novas
    df['education_unknown'] = [int(x=='unknown') for x in df['education']]
    df['education_secondary'] = [int(x=='secondary') for x in df['education']]
    df['education_primary'] = [int(x=='primary') for x in df['education']]
    df['education_tertiary'] = [int(x=='tertiary') for x in df['education']]
    df = df.drop('education', 1)
    #df['education'] = df['education'].replace(["unknown","secondary","primary","tertiary"], [0, 2, 1, 3])

    '''df['education_unknown'] =[]
    df['education_secondary'] = []
    df['education_primary'] = []
    df['education_tertiary'] = []
    for x in df['education']:
        df['education_unknown'].append(int(x=='unknown'))
        df['education_secondary'].append(int(x=='secondary') )
        df['education_primary'].append(int(x=='primary'))
        df['education_tertiary'].append(int(x=='tertiary'))
    df = df.drop('education', 1)'''
    
    # default
    df['default'] = df['default'].replace(["yes","no"], [1, 0])

    # housing
    df['housing'] = df['housing'].replace(["yes","no"], [1, 0])
    # loan
    df['loan'] = df['loan'].replace(["yes","no"], [1, 0])

    # contact
    # 3 novas
    df['contact_unknown'] = [int(x=='unknown') for x in df['contact']]
    df['contact_telephone'] = [int(x=='telephone') for x in df['contact']]
    df['contact_cellular'] = [int(x=='cellular') for x in df['contact']]
    df = df.drop('contact', 1)

    #poutcome
    # 4 novas
    df['poutcome_unknown'] = [int(x=='unknown') for x in df['poutcome']]
    df['poutcome_other'] = [int(x=='other') for x in df['poutcome']]
    df['poutcome_failure'] = [int(x=='failure') for x in df['poutcome']]
    df['poutcome_success'] = [int(x=='success') for x in df['poutcome']]
    df = df.drop('poutcome', 1)


    '''df['poutcome_unknown'].append(int(x=='unknown'))
    df['poutcome_other'].append(int(x=='other'))
    df['poutcome_failure'].append(int(x=='failure') )
    df['poutcome_success'].append(int(x=='success'))
    for x in df['poutcome']:
        df['poutcome_unknown'].append(int(x=='unknown'))
        df['poutcome_other'].append(int(x=='other'))
        df['poutcome_failure'].append(int(x=='failure'))
        df['poutcome_success'].append(int(x=='success'))
    df = df.drop('poutcome', 1)'''
    #df['poutcome'] = df['poutcome'].replace(['failure','unknown','other','success'], [0,1,2,3])

    # month
    # 12 novas
    df['month_jan'] = [int(x=='jan') for x in df['month']]
    df['month_fev'] = [int(x=='feb') for x in df['month']]
    df['month_mar'] = [int(x=='mar') for x in df['month']]
    df['month_abr'] = [int(x=='apr') for x in df['month']]
    df['month_mai'] = [int(x=='may') for x in df['month']]
    df['month_jun'] = [int(x=='jun') for x in df['month']]
    df['month_jul'] = [int(x=='jul') for x in df['month']]
    df['month_ago'] = [int(x=='aug') for x in df['month']]
    df['month_set'] = [int(x=='sep') for x in df['month']]
    df['month_out'] = [int(x=='oct') for x in df['month']]
    df['month_nov'] = [int(x=='nov') for x in df['month']]
    df['month_dez'] = [int(x=='dec') for x in df['month']]
    df = df.drop('month', 1)


    '''df['month_jan'] = []
    df['month_fev'] = []
    df['month_mar'] = []
    df['month_abr'] = []
    df['month_mai'] = []
    df['month_jun'] = []
    df['month_jul'] = []
    df['month_ago'] = []
    df['month_set'] = []
    df['month_out'] = []
    df['month_nov'] = []
    df['month_dez'] = []
    for x in df['month']:
        df['month_jan'].append(int(x=='jan'))
        df['month_fev'].append(int(x=='feb'))
        df['month_mar'].append(int(x=='mar'))
        df['month_abr'].append(int(x=='apr'))
        df['month_mai'].append(int(x=='may'))
        df['month_jun'].append(int(x=='jun'))
        df['month_jul'].append(int(x=='jul'))
        df['month_ago'].append(int(x=='aug'))
        df['month_set'].append(int(x=='sep'))
        df['month_out'].append(int(x=='oct'))
        df['month_nov'].append(int(x=='nov'))
        df['month_dez'].append(int(x=='dec'))


    df = df.drop('month', 1)'''
    df['y'] = df['y'].replace(['yes', 'no'], [1, 0])

    # possui os dados
    x = np.array(df.drop('y',1)) # axis=0 row e axis=1 columns
    # possui as classes
    y = np.array(df.y)

    return [x, y, 'Bank']

def dorothea():
    df = pd.read_csv('bases/dorothea/')


def waveform():
    # nao eh necessario colocar as colunas, estao sendo setadas com numeros, como ocorre com as ids
    df = pd.read_csv('bases/waveform/waveform-+noise.data', sep=',', header=None)

    df.columns = [str(x) for x in range(40)] + ["Class"]

    x = np.array(df.drop("Class", axis=1))
    y = np.array(df["Class"])
    
    return [x, y, 'Waveform', df.drop("Class", axis=1).columns]


    
    
    
    
    
    
    
    
    
    
