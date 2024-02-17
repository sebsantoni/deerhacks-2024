CREATE TABLE IF NOT EXISTS Ingredients (
    name VARCHAR PRIMARY KEY,


);

CREATE TABLE IF NOT EXISTS Recipes (
    id INT AUTO_INCREMENT PRIMARY KEY,
    Title VARCHAR,
    Instructions VARCHAR,
    Img VarBinary(max)
);