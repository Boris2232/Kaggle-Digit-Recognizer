def training_and_evaluating(model, optimizer, loss_type, device):
    for epoch in range(1, epochs + 1):

        print(f'Epoch {epoch} from {epochs}')

        def draw_plots(train_accuracy, validation_accuracy, train_loss, validation_loss):
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))

            axes[0][0].plot(train_accuracy, color = 'blue', linestyle = '--', label = 'Train Accuracy')
            axes[0][0].set_ylabel('Accuracy')
            axes[0][0].set_xticklabels([])
            axes[0][0].legend()

            axes[0][1].plot(validation_accuracy, color = 'red', linestyle = '--', label = 'Validation Accuracy')
            axes[0][1].set_ylabel('Accuracy')
            axes[0][1].set_xticklabels([])
            axes[0][1].legend()

            axes[1][0].plot(train_loss, color = 'blue', linestyle = '--', label = 'Train Loss')
            axes[1][0].set_ylabel('Loss')
            axes[1][0].set_xticklabels([])
            axes[1][0].legend()

            axes[1][1].plot(validation_loss, color = 'red', linestyle = '--', label = 'Validation Loss')
            axes[1][1].set_ylabel('Loss')
            axes[1][1].set_xticklabels([])
            axes[1][1].legend()
            plt.show()

        def print_epoch_results(engine, type=''):
            result = "\t".join([f"{i}:{j}" for i, j in engine.state.metrics.items()])
            print(f"Results after {type} epoch:", result)

        def one_epoch(engine):
            # evaluate on trainig and validation datasets
            train_eval.run(train_loader)
            valid_eval.run(validation_loader)

            # extract metrics from engine
            validation_results = valid_eval.state.metrics
            train_results = train_eval.state.metrics

            # metrics are stored into dictionary
            train_accuracy.append(round(train_results['Accuracy'], 4))
            validation_accuracy.append(round(validation_results['Accuracy'], 4))
            train_loss.append(train_results['Loss'])
            validation_loss.append(validation_results['Loss'])

            # dynamically sketch plots
            draw_plots(train_accuracy, validation_accuracy,
                       train_loss, validation_loss)

        # clear the grid before new epoch
        def clear_output_handler(engine):
            clear_output(wait=True)

        trainer = create_supervised_trainer(model, optimizer, loss_type, device)

        # choose metrics to show
        metrics_for_task = {
            "Accuracy": Accuracy(),
            # "Precision": Precision().mean(),
            # "Recall": Recall().mean(),
            "Loss": Loss(loss_type)
        }

        train_eval = create_supervised_evaluator(model, device=device, metrics=metrics_for_task)
        valid_eval = create_supervised_evaluator(model, device=device, metrics=metrics_for_task)

        # clear graph after epoch
        trainer.add_event_handler(Events.EPOCH_COMPLETED, clear_output_handler)

        # start new epoch
        trainer.add_event_handler(Events.EPOCH_COMPLETED, one_epoch)


        # print train and validation results
        train_eval.add_event_handler(Events.EPOCH_COMPLETED, print_epoch_results, type="Train")
        valid_eval.add_event_handler(Events.EPOCH_COMPLETED, print_epoch_results, type="Validation")

        # train model
        trainer.run(train_loader)