import time
from get_protein_kmer import read_fasta_file
from selenium import webdriver
from selenium.webdriver.support.ui import Select

def openChromBack():
    option = webdriver.ChromeOptions()
    option.add_argument('headless') # 设置option,可以在项目完成后再开启这个，注释这行方便调试
    wd = webdriver.Chrome(executable_path='C:/Users/yuhan/Downloads/chromedriver.exe',options=option) # 设置chromedriver.exe 和 option
    wd.get('https://npsa-prabi.ibcp.fr/cgi-bin/npsa_automat.pl?page=npsa_sopma.html')
    return wd


def operationAuth(driver,content):
    #输入序列
    elem = driver.find_element_by_name('notice')
    elem.send_keys(content)

    select = Select(driver.find_element_by_name('states'))
    # 先去掉所有选择的项
    #select.deselect_all()
    # 然后选择
    select.select_by_visible_text('3 (Helix, Sheet, Coil)')

    # 提交表单
    driver.find_element_by_xpath('/html/body/form/p[3]/input[1]').click()
    time.sleep(100)
    code = driver.find_elements_by_xpath('/html/body/pre[1]/code/font')
    seq_structure = []
    for i,char in enumerate(code):
        seq_structure.append(char.text)
    assert(len(seq_structure)==len(content))
    return ''.join(seq_structure)


if __name__ == '__main__':

    struc_map = {}
    seq_map = read_fasta_file('data/generated_data/RPI7317/protein_extracted_seq.fasta')
    i=0
    for name,seq in seq_map.items():
        i+=1
        print(i)
        print(name)
        driver = openChromBack()
        struc = operationAuth(driver,seq)
        struc_map[name] = struc
        print(struc)
        with open('data/generated_data/RPI7317/protein_structure_seq.fasta','a')as f:
            f.write(">"+name+"\n")
            f.write(struc+"\n")
        print("save")




