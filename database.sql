CREATE DATABASE IF NOT EXISTS aimhes_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE aimhes_db;

CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    guardian_email VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    dropout_level ENUM('Low', 'Medium', 'High'),
    dropout_prob DECIMAL(5,2),
    stress_level ENUM('No Stress', 'Eustress', 'Distress'),
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);