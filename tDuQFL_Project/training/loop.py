# training/loop.py
from common.imports import *
from ml.models import initialize_model
from ml.optimizers import make_spsa
from ml import optimizers as mlopt
from training.callbacks import reset_state, reset_callback_graph, learning_rates, perturbations, objective_func_vals
# NEW (correct source)
from ml.optimizers import objective_func_vals
from io_utils.csv_logger import save_results, save_best_client_results, save_accuracies_to_csv
from tele.teleport import teleport_parameters
from tele.noise import make_backend
from training.metrics import (
    metrics_init,
    metrics_log_round,
    metrics_finalize,
)


def compute_validation_loss(model, X, y):
    """
    Cross-entropy on (X, y). Falls back to NN.forward if predict_proba absent.
    """
    try:
        proba = model.predict_proba(X)
    except AttributeError:
        weights = model._fit_result.x
        proba = model._neural_network.forward(X, weights)
    return log_loss(y, proba)

def get_accuracy(model, test_sequences, test_labels):
    return model.score(test_sequences, test_labels)

def train_qnn_model(client_data, client_test_data, num_deep_unfolding_iterations,
                    initial_learning_rate, initial_perturbation, num_features,
                    csv_file, client_id=None, layer=None):
    initial_params = np.random.rand(RealAmplitudes(len(client_data[0]["sequence"]), reps=3).num_parameters)

    opt = make_spsa(
        maxiter=25,
        initial_learning_rate=initial_learning_rate,
        initial_perturbation=initial_perturbation,
        lr_alpha=0.01
    )
    model, opt = initialize_model(num_features, initial_params,
                                  initial_learning_rate, initial_perturbation,
                                  optimizer=opt)

    train_sequences = np.array([d["sequence"] for d in client_data])
    train_labels    = np.array([d["label"]    for d in client_data])
    test_sequences  = np.array([d["sequence"] for d in client_test_data])
    test_labels     = np.array([d["label"]    for d in client_test_data])

    total_time = 0.0

    for i in range(num_deep_unfolding_iterations):
        print("\n")
        print(f"Deep Unfolding Iteration {i+1}/{num_deep_unfolding_iterations}")

        # --- before fit: values that WILL be used now
        lr_used   = float(opt.learning_rate)
        pert_used = float(opt.perturbation)
        base_idx_lr   = len(mlopt.learning_rates)
        base_idx_pert = len(mlopt.perturbations)
        base_cb_calls = mlopt.callback_calls

        # per-fit normalization (approximate): 1 / maxiter
        mlopt.set_update_scale(1.0 / max(1, getattr(opt, "maxiter", 25)))

        t0 = time.time()
        # ONE FIT per unfolding step
        model.fit(train_sequences, train_labels)
        total_time += time.time() - t0

        # info for debugging
        cb_this_fit = mlopt.callback_calls - base_cb_calls
        #print(f"[client {client_id} | unfold {i+1}] callbacks this fit = {cb_this_fit}")

        # --- after fit: adopt only updates produced during THIS fit
        lr_seq   = mlopt.learning_rates[base_idx_lr:]
        pert_seq = mlopt.perturbations[base_idx_pert:]
        lr_next   = float(lr_seq[-1])   if lr_seq   else lr_used
        pert_next = float(pert_seq[-1]) if pert_seq else pert_used

        # trust-region caps (same thresholds as in optimizer module)
        lr_next   = float(np.clip(lr_next,   mlopt.LR_MIN,   mlopt.LR_MAX))
        pert_next = float(np.clip(pert_next, mlopt.PERT_MIN, mlopt.PERT_MAX))

        # push NEXT values into the live optimizer (for the next unfolding step)
        opt.learning_rate = lr_next
        opt.perturbation  = pert_next

        print(f"[client {client_id} | unfold {i+1}] "
              f"LR_used={lr_used:.6f}, PERT_used={pert_used:.6f} → "
              f"LR_next={lr_next:.6f}, PERT_next={pert_next:.6f}")
        

        # metrics
        train_acc = model.score(train_sequences, train_labels)
        test_acc  = model.score(test_sequences,  test_labels)
        obj_val   = mlopt.objective_func_vals[-1] if mlopt.objective_func_vals else None

        # CSV: log the values actually used on THIS unfold step
        save_results(csv_file, layer, client_id, i+1, obj_val, train_acc, test_acc,
                     lr_used, pert_used)

        # continuity of weights
        model.initial_point = model.weights

        print(f"Training Accuracy: {train_acc:.2f}")
        print(f"Test Accuracy: {test_acc:.2f}")

    # expose finals so caller can carry to next round if desired
    lr_final   = float(opt.learning_rate)
    pert_final = float(opt.perturbation)
    return model, train_acc, test_acc, total_time, lr_final, pert_final


def run_federated_training(*,
    clients, num_federated_layers, num_deep_unfolding_iterations,
    initial_learning_rate, initial_perturbation, num_features,
    best_client_csv_file, global_csv_file, local_csv_file, validation_csv_file,
    test_sequences, test_labels, X_val, y_val,
    use_teleportation=False, noise_preset='med', shots_used=256,
    metrics=None
):
    if metrics is None:
        raise ValueError("run_federated_training: 'metrics' store must be passed.")

    # per-client carry-over table
    client_hparams = [
        {"lr": float(initial_learning_rate), "pert": float(initial_perturbation)}
        for _ in range(len(clients))
    ]

    global_model_accuracy = []
    clients_train_accuracies, clients_test_accuracies = [], []
    round_times = []

    from io_utils.csv_logger import init_best_csv
    init_best_csv(best_client_csv_file)

    validation_loss_per_round = []
    info_last = None

    for epoch in tqdm(range(num_federated_layers), desc="Training Progress", leave=True):
        round_start = time.time()
        epoch_train_accuracies, epoch_test_accuracies = [], []
        best_client_index = -1
        best_client_accuracy = -1.0
        best_client_model = None
        print(f"[Round {epoch}] Teleportation {'ON' if use_teleportation else 'OFF'}")

        for index, client in enumerate(clients):
            # reset trackers per client (does NOT touch the carry table)
            reset_state()

            try:
                current_data = client.data[epoch]
            except IndexError:
                continue

            lr0   = client_hparams[index]["lr"]
            pert0 = client_hparams[index]["pert"]

            model, train_score, test_score, _, lr_final, pert_final = train_qnn_model(
                current_data, client.test_data,
                num_deep_unfolding_iterations,
                lr0, pert0, num_features,
                csv_file=local_csv_file, client_id=index, layer=epoch
            )

            # store finals to seed NEXT round for this client
            client_hparams[index]["lr"]   = lr_final
            client_hparams[index]["pert"] = pert_final

            epoch_train_accuracies.append(train_score)
            epoch_test_accuracies.append(test_score)

            if test_score > best_client_accuracy:
                best_client_accuracy = test_score
                best_client_index = index
                best_client_model = model

        round_duration = time.time() - round_start
        round_times.append(round_duration)
        save_best_client_results(best_client_csv_file, epoch, best_client_index, round_duration)

        # Global model ← best client, (optional teleport)
        global_model = best_client_model
        global_params = global_model.weights

        if use_teleportation:
            backend = make_backend(noise=noise_preset, shots=shots_used)
            updated_params, info = teleport_parameters(
                global_params, mode='perturb', noise=noise_preset, shots=shots_used,
                backend=backend, show_histogram=False, apply_shrinkage=True
            )
            info_last = info
        else:
            updated_params = np.asarray(global_params, dtype=np.float32)
            info_last = None

        # broadcast
        global_model.initial_point = updated_params
        for client in clients:
            client.primary_model = global_model

        # evaluate + log
        global_accuracy = global_model.score(test_sequences, test_labels)
        global_model_accuracy.append(global_accuracy)
        clients_train_accuracies.append(epoch_train_accuracies)
        clients_test_accuracies.append(epoch_test_accuracies)

        L_val = compute_validation_loss(global_model, X_val, y_val)
        metrics_log_round(
            metrics,
            round_idx=epoch,
            acc_global=global_accuracy,
            client_accs=epoch_test_accuracies,
            time_s=round_duration,
            val_loss=L_val,
            use_teleportation=use_teleportation,
            info=info_last if use_teleportation else None,
            noise=noise_preset if use_teleportation else None,
            shots=shots_used if use_teleportation else None,
        )
        validation_loss_per_round.append(L_val)
        with open(validation_csv_file, mode='a', newline='') as f:
            csv.writer(f).writerow([epoch, L_val])

        # reset ONLY callback trackers (do NOT touch client_hparams)
        # If you truly want to reinitialize learnability each new round, you could
        # overwrite client_hparams here; otherwise keep them to accumulate across rounds.
        # (Your previous reset_callback_graph(initial_lr, initial_pert) is no longer needed.)

        save_accuracies_to_csv(global_model_accuracy, clients_train_accuracies,
                               clients_test_accuracies, filename=global_csv_file)

    return (global_model_accuracy, clients_train_accuracies, clients_test_accuracies,
            round_times, validation_loss_per_round, info_last)
