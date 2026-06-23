import os
import re
import sqlite3
import json
import logging
import ast
import numpy as np
logger = logging.getLogger(__name__)

def weighted_gaussian_by_mad(probs, k=2.5):
    med = np.median(probs)
    mad = np.median(np.abs(probs - med))  # MAD
    if mad == 0:
        return med
    # distance in MAD units
    d = np.abs(probs - med) / mad
    # weight = exp(-(d/k)**2)  where k controls cutoff
    w = np.exp(-(d / k)**2)
    return (w * probs).sum() / w.sum()

class EXPERIMENTS_DB:
    def __init__(self, path="experiments.db"):
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.cur = self.conn.cursor()

    def _create_schema(self):
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS Experiment (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        codename TEXT NOT NULL UNIQUE
        );
        """)

        self.cur.execute("""
        CREATE TABLE Dataset (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        codename TEXT NOT NULL UNIQUE
        );
        """)

        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS ExperimentDataset (
        experiment_id INTEGER NOT NULL,
        dataset_id INTEGER NOT NULL,
        PRIMARY KEY (experiment_id, dataset_id),
        FOREIGN KEY (experiment_id) REFERENCES Experiment(id) ON DELETE CASCADE,
        FOREIGN KEY (dataset_id) REFERENCES Dataset(id) ON DELETE CASCADE
        );""")

        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS Sample (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_id INTEGER NOT NULL,
        file_path TEXT NOT NULL UNIQUE,
        label INTEGER NOT NULL CHECK (label in (0, 1)),
        FOREIGN KEY (dataset_id) REFERENCES Dataset(id) ON DELETE CASCADE
        );
        """)

        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS ModelRun (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sample_id INTEGER NOT NULL,
        experiment_id INTEGER NOT NULL,
        model_codename TEXT NOT NULL CHECK (model_codename IN ('ee_cnn', 'vit4v', 'wifi')),
        exit_idx INTEGER,
        scores BLOB,
        FOREIGN KEY (sample_id) REFERENCES Sample(id) ON DELETE CASCADE,
        FOREIGN KEY (experiment_id) REFERENCES Experiment(id) ON DELETE CASCADE
        );
        """)

        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS Performance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        modelrun_id INTEGER NOT NULL,
        phase TEXT NOT NULL CHECK (phase IN ('preprocessing','inference', 'transferring')),
        duration REAL NOT NULL,
        tot_energy REAL NOT NULL,
        cpu_energy REAL NOT NULL,
        gpu_energy REAL NOT NULL,
        ram_energy REAL NOT NULL,
        FOREIGN KEY (modelrun_id) REFERENCES ModelRun(id) ON DELETE CASCADE,
        UNIQUE(modelrun_id, phase)
        );
        """)

        self.conn.commit()

        logger.info("Successfully initialized the experiments database.")

    def reset_database(self):
        self.cur.execute("PRAGMA foreign_keys = OFF;")
        
        tables = ["Performance", "ModelRun", "Sample", "ExperimentDataset", "Experiment", "Dataset"]
        for t in tables:
            self.cur.execute(f"DROP TABLE IF EXISTS {t};")

        self.conn.commit()
        self.cur.execute("PRAGMA foreign_keys = ON;")
        self._create_schema()
        
        logger.info("Database fully reset and recreated.")
        
    def add_experiment(self, codename):
        try:
            self.cur.execute(
                "INSERT INTO Experiment (codename) VALUES (?)",
                (codename,)
            )
        except sqlite3.IntegrityError as e:
            logger.info(f"Experiment {codename} already exists. Skipping.")
            return 0
        self.conn.commit()
        return 1

    def add_dataset(self, codename):
        try:
            self.cur.execute(
                "INSERT INTO Dataset (codename) VALUES (?)",
                (codename,)
            )
        except sqlite3.IntegrityError as e:
            logger.info(f"Dataset {codename} already exists. Skipping.")
            return 0
        self.conn.commit()
        return 1

    def add_dataset_to_experiment(self, experiment_codename, dataset_codename):
        experiment_id = self.get_experiment_id(experiment_codename)
        dataset_id = self.get_dataset_id(dataset_codename)
        
        self.cur.execute("""
            INSERT OR IGNORE INTO ExperimentDataset (experiment_id, dataset_id)
            VALUES (?, ?)
        """, (experiment_id, dataset_id))
        self.conn.commit()

    def add_sample(self, dataset_codename, file_path, label):
        dataset_id = self.get_dataset_id(dataset_codename)

        self.cur.execute(
            "INSERT INTO Sample (dataset_id, file_path, label) VALUES (?, ?, ?)",
            (dataset_id, file_path, label)
        )
        self.conn.commit()
        return self.cur.lastrowid

    def add_modelrun(self, sample_file_path, experiment_codename, model_codename, exit_idx, scores):
        sample_id = self.get_sample_id(sample_file_path)
        experiment_id = self.get_experiment_id(experiment_codename)
        scores_json = json.dumps(scores)
        self.cur.execute(
            "INSERT INTO ModelRun (sample_id, experiment_id, model_codename, exit_idx, scores) VALUES (?, ?, ?, ?, ?)",
            (sample_id, experiment_id, model_codename, exit_idx, scores_json)
        )
        self.conn.commit()
        return self.cur.lastrowid

    def add_performance(self, modelrun_id, phase, duration, tot_energy, cpu_energy, gpu_energy, ram_energy):
        self.cur.execute(
            "INSERT INTO Performance (modelrun_id, phase, duration, tot_energy, cpu_energy, gpu_energy, ram_energy) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (modelrun_id, phase, duration, tot_energy, cpu_energy, gpu_energy, ram_energy)
        )
        self.conn.commit()
        return self.cur.lastrowid

    def add_modelrun_with_performance(
        self,
        sample_file_path,
        experiment_codename,
        model_codename,
        exit_idx,
        scores,
        preprocessing_perf,
        inference_perf
    ):
        """
        preprocessing_perf = (duration, consumption)
        inference_perf     = (duration, consumption)
        """
        modelrun_id = self.add_modelrun(sample_file_path, experiment_codename,
                                        model_codename, exit_idx, scores)

        self.add_performance(
            modelrun_id,
            "preprocessing",
            *preprocessing_perf
        )

        self.add_performance(
            modelrun_id,
            "inference",
            *inference_perf
        )

        return modelrun_id

    def add_transferring_modelrun_with_performance(
        self,
        sample_file_path,
        experiment_codename,
        model_codename,
        exit_idx,
        scores,
        transferring_perf
    ):
        modelrun_id = self.add_modelrun(sample_file_path, experiment_codename,
                                        model_codename, exit_idx, scores)

        self.add_performance(
            modelrun_id,
            "transferring",
            *transferring_perf
        )

        return modelrun_id

    def get_dataset_id(self, codename):
        self.cur.execute("SELECT id FROM Dataset WHERE codename=?", (codename,))
        dataset = self.cur.fetchone()
        if not dataset:
            raise ValueError(f"Dataset '{codename}' not found.")
        return dataset[0]

    def get_experiment_id(self, codename):
        self.cur.execute("SELECT id FROM Experiment WHERE codename=?", (codename,))
        experiment = self.cur.fetchone()
        if not experiment:
            raise ValueError(f"Experiment '{codename}' not found.")
        return experiment[0]

    def get_sample_id(self, file_path):
        self.cur.execute("SELECT id FROM Sample WHERE file_path=?", (file_path,))
        sample = self.cur.fetchone()
        if not sample:
            raise ValueError(f"Sample with file path '{file_path}' not found.")
        return sample[0]

    def get_dataset_experiments(self, dataset_codename):
        ds_id = self.get_dataset_id(dataset_codename)
        
        self.cur.execute("""
        SELECT e.codename
        FROM Experiment e
        JOIN ExperimentDataset ed ON e.id = ed.experiment_id
        JOIN Dataset d ON d.id = ed.dataset_id
        WHERE d.id=?
        """, (ds_id,))
        return self.cur.fetchall()
    
    def get_experiment_runs(self, codename):
        self.cur.execute("""
        SELECT s.file_path, m.model_codename, m.scores, m.exit_idx, p.phase, p.duration, p.tot_energy, p.cpu_energy, p.gpu_energy, p.ram_energy
        FROM Experiment e
        JOIN ExperimentDataset ed ON e.id = ed.experiment_id
        JOIN Dataset d ON d.id = ed.dataset_id
        JOIN Sample s ON s.dataset_id = d.id
        JOIN ModelRun m ON m.sample_id = s.id
        JOIN Performance p ON p.modelrun_id = m.id
        WHERE e.codename = (?)
        """, (codename,))
        return self.cur.fetchall()

    def get_experiment_runs_w_modelrun(self, codename):
        self.cur.execute("""
        SELECT s.file_path, m.id, m.model_codename, m.scores, m.exit_idx, p.phase, p.duration, p.tot_energy, p.cpu_energy, p.gpu_energy, p.ram_energy
        FROM Experiment e
        JOIN ExperimentDataset ed ON e.id = ed.experiment_id
        JOIN Dataset d ON d.id = ed.dataset_id
        JOIN Sample s ON s.dataset_id = d.id
        JOIN ModelRun m ON m.sample_id = s.id
        JOIN Performance p ON p.modelrun_id = m.id
        WHERE e.codename = (?)
        """, (codename,))
        return self.cur.fetchall()    

    def get_dataset_samples(self, dataset_codename):
        self.cur.execute("""
        SELECT s.label, s.file_path
        FROM DATASET d
        JOIN Sample s ON s.dataset_id = d.id
        WHERE d.codename = (?)
        """, (dataset_codename,))
        return self.cur.fetchall()

    def get_experiments_datasets(self):
        self.cur.execute("""
        SELECT d.codename, e.codename
        FROM ExperimentDataset ed
        JOIN Experiment e ON e.id = ed.experiment_id
        JOIN Dataset d ON d.id = ed.dataset_id
        """)
        return self.cur.fetchall()

    def get_scores(self, experiment_codename:str, dataset_codename:str, model_codename:str, exit_idx:int|None=None):
        self.cur.execute("""
        SELECT m.scores
        FROM ExperimentDataset ed
        JOIN Experiment e ON e.id = ed.experiment_id
        JOIN Dataset d ON d.id = ed.dataset_id
        JOIN Sample s ON s.dataset_id = d.id
        JOIN ModelRun m ON m.sample_id = s.id
        WHERE e.codename = ?
        AND d.codename = ?
        AND m.model_codename = ?
        AND (m.exit_idx = ? OR m.exit_idx IS NULL)
        ORDER BY s.id
        """, (experiment_codename, dataset_codename, model_codename, exit_idx))
        rows = self.cur.fetchall()
        scores = [ast.literal_eval(row[0]) for row in rows]
        return scores

    def get_performance(self, experiment_codename:str, dataset_codename:str, model_codename:str, exit_idx:int|None=None):
        self.cur.execute("""
        SELECT
        SUM(CASE WHEN p.phase = 'preprocessing' THEN p.duration END) AS preprocessing_duration,
        SUM(CASE WHEN p.phase = 'preprocessing' THEN p.tot_energy END) AS preprocessing_tot_energy,
        SUM(CASE WHEN p.phase = 'preprocessing' THEN p.cpu_energy END) AS preprocessing_cpu_energy,
        SUM(CASE WHEN p.phase = 'preprocessing' THEN p.gpu_energy END) AS preprocessing_gpu_energy,
        SUM(CASE WHEN p.phase = 'preprocessing' THEN p.ram_energy END) AS preprocessing_ram_energy,
        SUM(CASE WHEN p.phase = 'inference' THEN p.duration END) AS inference_duration,
        SUM(CASE WHEN p.phase = 'inference' THEN p.tot_energy END) AS inference_tot_energy,
        SUM(CASE WHEN p.phase = 'inference' THEN p.cpu_energy END) AS inference_cpu_energy,
        SUM(CASE WHEN p.phase = 'inference' THEN p.gpu_energy END) AS inference_gpu_energy,
        SUM(CASE WHEN p.phase = 'inference' THEN p.ram_energy END) AS inference_ram_energy
        FROM ExperimentDataset ed
        JOIN Experiment e ON e.id = ed.experiment_id
        JOIN Dataset d ON d.id = ed.dataset_id
        JOIN Sample s ON s.dataset_id = d.id
        JOIN ModelRun m ON m.sample_id = s.id
        JOIN Performance p ON p.modelrun_id = m.id
        WHERE e.codename = ?
        AND d.codename = ?
        AND m.model_codename = ?
        AND (m.exit_idx = ? OR m.exit_idx IS NULL)
        GROUP BY m.id
        ORDER BY s.id
        """, (experiment_codename, dataset_codename, model_codename, exit_idx))
        rows = self.cur.fetchall()
        preprocessing = [list(row)[:5] for row in rows]
        inference = [list(row)[5:] for row in rows]
        performance = {
            "preprocessing": preprocessing,
            "inference": inference
        }
        
        return performance

    def get_transferring_performance(self, experiment_codename:str, dataset_codename:str, model_codename:str, exit_idx:int|None=None):
        self.cur.execute("""
        SELECT p.duration, p.tot_energy, p.cpu_energy, p.gpu_energy, p.ram_energy
        FROM ExperimentDataset ed
        JOIN Experiment e ON e.id = ed.experiment_id
        JOIN Dataset d ON d.id = ed.dataset_id
        JOIN Sample s ON s.dataset_id = d.id
        JOIN ModelRun m ON m.sample_id = s.id
        JOIN Performance p ON p.modelrun_id = m.id
        WHERE e.codename = ?
        AND d.codename = ?
        AND m.model_codename = ?
        AND (m.exit_idx = ? OR m.exit_idx IS NULL)
        AND p.phase = 'transferring'
        GROUP BY m.id
        ORDER BY s.id
        """, (experiment_codename, dataset_codename, model_codename, exit_idx))
        rows = self.cur.fetchall()
        
        return rows
    
    def get_labels(self, dataset_codename):
        self.cur.execute("""
        SELECT s.label
        FROM Dataset d
        JOIN Sample s ON s.dataset_id = d.id
        WHERE d.codename = ?
        ORDER BY s.id
        """, (dataset_codename,))
        rows = self.cur.fetchall()
        labels = [row[0] for row in rows]
        return labels

    def rename_experiment(self, old_codename, new_codename):
        try:
            self.cur.execute("""
                UPDATE Experiment
                SET codename = ?
                WHERE codename = ?
            """, (new_codename, old_codename))

            self.conn.commit()

        except sqlite3.IntegrityError:
            print(f"Codename '{new_codename}' already exists.")
    
    def load_vit_validation(self,
                            dataset_codename:str="vit_validation",
                            video_directory:str="/mnt/datasets/prin/video_raw/v2/"):
        res = self.add_dataset(dataset_codename)

        if not res:
            return

        pattern = r"video([0-9]+)_varroa_(free|infested)_[0-9]+-[0-9]+"
    
        file_path = "./vit_validation_partition.txt"
        
        with open(file_path, 'r') as f:
            content = f.read()

        matches = re.findall(pattern, content)
    
        testing_free_ids = set([int(m[0]) for m in matches if m[1] == "free"])
        testing_infested_ids = set([int(m[0]) for m in matches if m[1] == "infested"])
        
        id_pattern = r"^([0-9]+) [\w\- .]*\.mkv$"
        
        free_video_directory = video_directory + "varroa_free/"
        free_video_files = [f for f in sorted(os.listdir(free_video_directory))
                            if (os.path.isfile(os.path.join(free_video_directory, f)) and
                                f.endswith('.mkv'))]
        for free_video in free_video_files:
            match = re.match(id_pattern, free_video)
            video_id = int(match.group(1))
            if match and video_id in testing_free_ids:
                self.add_sample(dataset_codename, os.path.join(free_video_directory, free_video), 0)

        infested_video_directory = video_directory + "varroa_infested/"
        infested_video_files = [f for f in sorted(os.listdir(infested_video_directory))
                                if (os.path.isfile(os.path.join(infested_video_directory, f)) and
                                    f.endswith('.mkv'))]
        for infested_video in infested_video_files:
            match = re.match(id_pattern, infested_video)
            video_id = int(match.group(1))
            if match and video_id in testing_infested_ids:
                self.add_sample(dataset_codename, os.path.join(infested_video_directory, infested_video), 1)

    def close(self):
        self.conn.close()

        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    db = EXPERIMENTS_DB()
    """
    db.reset_database()
    db.load_vit_validation("vit_validation")

    db.add_experiment("experiment_3")
    db.add_dataset_to_experiment("experiment_3", "vit_validation")

    rows = db.get_dataset_samples("vit_validation")
    for label, file_path in rows:
        db.add_modelrun_with_performance(
            file_path,
            "experiment_3",
            "vit4v",
            None,
            scores=[0.89, 0.84, 0.74],
            preprocessing_perf=(1.05, 0.33),
            inference_perf=(0.80, 0.20)
        )

    rows = db.get_experiment_runs("experiment_3")
    print("Experiment runs")
    for row in rows:
        print(row)
    """
    """
    rows = db.get_experiment_runs("experiment_3")
    print("Experiment runs")
    for row in rows:
        print(row)
    """
    scores = db.get_scores("experiment_3", "vit_validation", "ee_cnn", 0)
    scores = db.get_scores("experiment_3", "vit_validation", "ee_cnn", 1)
    scores = db.get_scores("experiment_3", "vit_validation", "vit4v")

    performance = db.get_performance("experiment_3", "vit_validation", "ee_cnn", 0)
    performance = db.get_performance("experiment_3", "vit_validation", "ee_cnn", 1)
    performance = db.get_performance("experiment_3", "vit_validation", "vit4v")

    labels = db.get_labels("vit_validation")
    # print("vit_validation", labels)

    db.close()
