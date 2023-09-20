# Description of ADMET Prediction Datasets

> In a typical circumstance (0: non-activity; 1: activity).  In toxicity prediction(0: non-toxicity; 1: toxicity)

### Caco-2 (Cell Effective Permeability), Wang et al.

**Dataset Description:** The human colon epithelial cancer cell line, Caco-2, is used as an in vitro model to simulate the human intestinal tissue. The experimental result on the rate of drug passing through the Caco-2 cells can approximate the rate at which the drug permeates through the human intestinal tissue.

**Task Description:** Regression. Given a drug SMILES string, predict the Caco-2 cell effective permeability.



### HIA (Human Intestinal Absorption), Hou et al.

**Dataset Description:** When a drug is orally administered, it needs to be absorbed from the human gastrointestinal system into the bloodstream of the human body. This ability of absorption is called human intestinal absorption (HIA) and it is crucial for a drug to be delivered to the target.

**Task Description:** Binary classification. Given a drug SMILES string, predict the activity of HIA. (0: non-activity; 1: activity)



### Pgp (P-glycoprotein) Inhibition, Broccatelli et al.

**Dataset Description:** P-glycoprotein (Pgp) is an ABC transporter protein involved in intestinal absorption, drug metabolism, and brain penetration, and its inhibition can seriously alter a drug's bioavailability and safety. In addition, inhibitors of Pgp can be used to overcome multidrug resistance.

**Task Description:** Binary classification. Given a drug SMILES string, predict the activity of Pgp inhibition. (0: non-activity; 1: activity)



### Bioavailability, Ma et al.

**Dataset Description:** Oral bioavailability is defined as “the rate and extent to which the active ingredient or active moiety is absorbed from a drug product and becomes available at the site of action”.

**Task Description:** Binary classification. Given a drug SMILES string, predict the activity of bioavailability. (0: non-activity; 1: activity)



### Lipophilicity, AstraZeneca

**Dataset Description:** Lipophilicity measures the ability of a drug to dissolve in a lipid (e.g. fats, oils) environment. High lipophilicity often leads to high rate of metabolism, poor solubility, high turn-over, and low absorption. From MoleculeNet.

**Task Description:** Regression. Given a drug SMILES string, predict the activity of lipophilicity.



### Solubility, AqSolDB

**Dataset Description:** Aqeuous solubility measures a drug's ability to dissolve in water. Poor water solubility could lead to slow drug absorptions, inadequate bioavailablity and even induce toxicity. More than 40% of new chemical entities are not soluble.

**Task Description:** Regression. Given a drug SMILES string, predict the activity of solubility.



### BBB (Blood-Brain Barrier), Martins et al.

**Dataset Description:** As a membrane separating circulating blood and brain extracellular fluid, the blood-brain barrier (BBB) is the protection layer that blocks most foreign drugs. Thus the ability of a drug to penetrate the barrier to deliver to the site of action forms a crucial challenge in development of drugs for central nervous system From MoleculeNet.

**Task Description:** Binary classification. Given a drug SMILES string, predict the activity of BBB. (0: non-activity; 1: activity)



### PPBR (Plasma Protein Binding Rate), AstraZeneca

**Dataset Description:** The human plasma protein binding rate (PPBR) is expressed as the percentage of a drug bound to plasma proteins in the blood. This rate strongly affect a drug's efficiency of delivery. The less bound a drug is, the more efficiently it can traverse and diffuse to the site of actions. From a ChEMBL assay deposited by AstraZeneca.

**Task Description:** Regression. Given a drug SMILES string, predict the rate of PPBR.



### VDss (Volumn of Distribution at steady state), Lombardo et al.

**Dataset Description:** The volume of distribution at steady state (VDss) measures the degree of a drug's concentration in body tissue compared to concentration in blood. Higher VD indicates a higher distribution in the tissue and usually indicates the drug with high lipid solubility, low plasma protein binidng rate.

**Task Description:** Regression. Given a drug SMILES string, predict the volume of distributon.



### CYP P450 2C19 Inhibition, Veith et al.

**Dataset Description:** The CYP P450 genes are essential in the breakdown (metabolism) of various molecules and chemicals within cells. A drug that can inhibit these enzymes would mean poor metabolism to this drug and other drugs, which could lead to drug-drug interactions and adverse effects. Specifically, the CYP2C19 gene provides instructions for making an enzyme called the endoplasmic reticulum, which is involved in protein processing and transport.

**Task Description:** Binary Classification. Given a drug SMILES string, predict CYP2C19 inhibition.(0: non-activity; 1: activity)



### CYP P450 2D6 Inhibition, Veith et al.

**Dataset Description:** The CYP P450 genes are involved in the formation and breakdown (metabolism) of various molecules and chemicals within cells. Specifically, CYP2D6 is primarily expressed in the liver. It is also highly expressed in areas of the central nervous system, including the substantia nigra.

**Task Description:** Binary Classification. Given a drug SMILES string, predict CYP2D6 inhibition.(0: non-activity; 1: activity)



### CYP P450 3A4 Inhibition, Veith et al.

**Dataset Description:** The CYP P450 genes are involved in the formation and breakdown (metabolism) of various molecules and chemicals within cells. Specifically, CYP3A4 is an important enzyme in the body, mainly found in the liver and in the intestine. It oxidizes small foreign organic molecules (xenobiotics), such as toxins or drugs, so that they can be removed from the body.

**Task Description:** Binary Classification. Given a drug SMILES string, predict CYP3A4 inhibition.(0: non-activity; 1: activity)



### CYP P450 1A2 Inhibition, Veith et al.

**Dataset Description:** The CYP P450 genes are involved in the formation and breakdown (metabolism) of various molecules and chemicals within cells. Specifically, CYP1A2 localizes to the endoplasmic reticulum and its expression is induced by some polycyclic aromatic hydrocarbons (PAHs), some of which are found in cigarette smoke. It is able to metabolize some PAHs to carcinogenic intermediates. Other xenobiotic substrates for this enzyme include caffeine, aflatoxin B1, and acetaminophen.

**Task Description:** Binary Classification. Given a drug SMILES string, predict CYP1A2 inhibition.(0: non-activity; 1: activity)



### CYP P450 2C9 Inhibition, Veith et al.

**Dataset Description:** The CYP P450 genes are involved in the formation and breakdown (metabolism) of various molecules and chemicals within cells. Specifically, the CYP P450 2C9 plays a major role in the oxidation of both xenobiotic and endogenous compounds.

**Task Description:** Binary Classification. Given a drug SMILES string, predict CYP2C9 inhibition.(0: non-activity; 1: activity)



### CYP2C9 Substrate, Carbon-Mangels et al.

**Dataset Description:** CYP P450 2C9 plays a major role in the oxidation of both xenobiotic and endogenous compounds. Substrates are drugs that are metabolized by the enzyme. TDC used a dataset from [1], which merged information on substrates and nonsubstrates from six publications.

**Task Description:** Binary Classification. Given a drug SMILES string, predict if it is a substrate to the enzyme.(0: non-activity; 1: activity)

**References:**

[[1\] Carbon‐Mangels, Miriam, and Michael C. Hutter. “Selecting relevant descriptors for classification by bayesian estimates: a comparison with decision trees and support vector machines approaches for disparate data sets.” Molecular informatics 30.10 (2011): 885-895.](https://onlinelibrary.wiley.com/doi/abs/10.1002/minf.201100069?casa_token=an2_FKO2WnMAAAAA:d-W7F_Ukq1ArtZoSaqh96zcwyAQlVtzc9IR12M9WltU8U0iJJW_BLINGNbC9qN7dk7mFmsNIy1bVcZdt)



### CYP2D6 Substrate, Carbon-Mangels et al.

**Dataset Description:** CYP2D6 is primarily expressed in the liver. It is also highly expressed in areas of the central nervous system, including the substantia nigra. TDC used a dataset from [1] , which merged information on substrates and nonsubstrates from six publications.

**References:**

[[1\] Carbon‐Mangels, Miriam, and Michael C. Hutter. “Selecting relevant descriptors for classification by bayesian estimates: a comparison with decision trees and support vector machines approaches for disparate data sets.” Molecular informatics 30.10 (2011): 885-895.](https://onlinelibrary.wiley.com/doi/abs/10.1002/minf.201100069?casa_token=an2_FKO2WnMAAAAA:d-W7F_Ukq1ArtZoSaqh96zcwyAQlVtzc9IR12M9WltU8U0iJJW_BLINGNbC9qN7dk7mFmsNIy1bVcZdt)



### CYP3A4 Substrate, Carbon-Mangels et al.

**Dataset Description:** CYP3A4 is an important enzyme in the body, mainly found in the liver and in the intestine. It oxidizes small foreign organic molecules (xenobiotics), such as toxins or drugs, so that they can be removed from the body. TDC used a dataset from [1], which merged information on substrates and nonsubstrates from six publications.

**Task Description:** Binary Classification. Given a drug SMILES string, predict if it is a substrate to the enzyme.(0: non-activity; 1: activity)

**References:**

[[1\] Carbon‐Mangels, Miriam, and Michael C. Hutter. “Selecting relevant descriptors for classification by bayesian estimates: a comparison with decision trees and support vector machines approaches for disparate data sets.” Molecular informatics 30.10 (2011): 885-895.](https://onlinelibrary.wiley.com/doi/abs/10.1002/minf.201100069?casa_token=an2_FKO2WnMAAAAA:d-W7F_Ukq1ArtZoSaqh96zcwyAQlVtzc9IR12M9WltU8U0iJJW_BLINGNbC9qN7dk7mFmsNIy1bVcZdt)



### Half Life, Obach et al.

**Dataset Description:** Half life of a drug is the duration for the concentration of the drug in the body to be reduced by half. It measures the duration of actions of a drug. This dataset is from [1] and we obtain the deposited version under CHEMBL assay 1614674.

**Task Description:** Regression. Given a drug SMILES string, predict the half life duration.

**References:**

[[1\] Obach, R. Scott, Franco Lombardo, and Nigel J. Waters. “Trend analysis of a database of intravenous pharmacokinetic parameters in humans for 670 drug compounds.” Drug Metabolism and Disposition 36.7 (2008): 1385-1405.](http://dmd.aspetjournals.org/content/36/7/1385.short?casa_token=9RLY3ZV5uqwAAAAA:j6LPQLnRLDOdzsVkew-nT5eIk9i_6LfkRf6FaBkIS1UKDe7oqP2NmymGeSRsGYigtTQXVHgXyoek)



### Clearance, AstraZeneca

**Dataset Description:** Drug clearance is defined as the volume of plasma cleared of a drug over a specified time period and it measures the rate at which the active drug is removed from the body. This is a dataset curated from ChEMBL database containing experimental results on intrinsic clearance, deposited from AstraZeneca. It contains clearance measures from two experiments types, hepatocyte and microsomes. As many studies [2] have shown various clearance outcomes given these two different types, we separate them.

**Task Description:** Regression. Given a drug SMILES string, predict the activity of clearance.

**References:**

[[1\] AstraZeneca. Experimental in vitro Dmpk and physicochemical data on a set of publicly disclosed compounds (2016)](https://doi.org/10.6019/chembl3301361)

[[2\] Di, Li, et al. “Mechanistic insights from comparing intrinsic clearance values between human liver microsomes and hepatocytes to guide drug design.” European journal of medicinal chemistry 57 (2012): 441-448.](https://www.sciencedirect.com/science/article/pii/S0223523412003959?casa_token=MNBb1xkUT_YAAAAA:3s1kw30Nlv_tSrXYURqsz4j7it5wswWolxgkAPOg_Ts-FqVv3jLnyUFRQu9rNEdyV2FyyXL0VxI)



### Acute Toxicity LD50

**Dataset Description:** Acute toxicity LD50 measures the most conservative dose that can lead to lethal adverse effects. The higher the dose, the more lethal of a drug. This dataset is kindly provided by the authors of [1].

**Task Description:** Regression. Given a drug SMILES string, predict its acute toxicity.

**References:**

[[1\] Zhu, Hao, et al. “Quantitative structure− activity relationship modeling of rat acute toxicity by oral exposure.” Chemical research in toxicology 22.12 (2009): 1913-1921.](https://pubs.acs.org/doi/abs/10.1021/tx900189p?casa_token=vfBbuxuUCqEAAAAA:YAcI0r4Z3rtlRYP_l5H8OlTfdUh3DVlO6ws_h1NkhpaXH3-NrdI2-s5ghWWJbxfPQw-KhQIAwMi1Di3v)



### hERG blockers

**Dataset Description:** Human ether-à-go-go related gene (hERG) is crucial for the coordination of the heart's beating. Thus, if a drug blocks the hERG, it could lead to severe adverse effects. Therefore, reliable prediction of hERG liability in the early stages of drug design is quite important to reduce the risk of cardiotoxicity-related attritions in the later development stages.

**Task Description:** Binary classification. Given a drug SMILES string, predict whether it blocks (1) or not blocks (0).



### Ames Mutagenicity

**Dataset Description:** Mutagenicity means the ability of a drug to induce genetic alterations. Drugs that can cause damage to the DNA can result in cell death or other severe adverse effects. Nowadays, the most widely used assay for testing the mutagenicity of compounds is the Ames experiment which was invented by a professor named Ames. The Ames test is a short-term bacterial reverse mutation assay detecting a large number of compounds which can induce genetic damage and frameshift mutations. The dataset is aggregated from four papers.

**Task Description:** Binary classification. Given a drug SMILES string, predict whether it is mutagenic (1) or not mutagenic (0).



### DILI (Drug Induced Liver Injury)

**Dataset Description:** Drug-induced liver injury (DILI) is fatal liver disease caused by drugs and it has been the single most frequent cause of safety-related drug marketing withdrawals for the past 50 years (e.g. iproniazid, ticrynafen, benoxaprofen). This dataset is aggregated from U.S. FDA’s National Center for Toxicological Research.

**Task Description:** Binary classification. Given a drug SMILES string, predict whether it can cause liver injury (1) or not (0).



### Skin Reaction

**Dataset Description:** Repetitive exposure to a chemical agent can induce an immune reaction in inherently susceptible individuals that leads to skin sensitization. The dataset used in this study was retrieved from the ICCVAM (Interagency Coordinating Committee on the Validation of Alternative Methods) report on the rLLNA.

**Task Description:** Binary classification. Given a drug SMILES string, predict whether it can cause skin reaction (1) or not (0).



### Carcinogens

**Dataset Description:** A carcinogen is any substance, radionuclide, or radiation that promotes carcinogenesis, the formation of cancer. This may be due to the ability to damage the genome or to the disruption of cellular metabolic processes.

**Task Description:** Binary classification. Given a drug SMILES string, predict whether it can cause carcinogen (1) or not (0).



### ClinTox

**Dataset Description:** The ClinTox dataset includes drugs that have failed clinical trials for toxicity reasons and also drugs that are associated with successful trials.

**Task Description:** Binary classification. Given a drug SMILES string, predict the clinical toxicity  (1) or not (0).



