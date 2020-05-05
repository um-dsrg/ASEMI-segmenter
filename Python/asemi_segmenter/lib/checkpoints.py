'''
Module for the checkpoint manager.

A checkpoint is a point in command's process that can be skipped if it has already been reached
before. For example if a file is to be created but has already been created in a previous run
(which was interrupted) then it can be checkpointed so that its creation will be skipped in a
subsequent run.

A checkpoint is a dictionary where keys are unique names for checkpoints in a command's process
and values are the number of times that the checkpoint was reached. A checkpoint file is a JSON
encoded file consisting of a dictionary of checkpoints. The upper dictionary's keys are the name
of the command being processed and the values are checkpoint dictionaries of those commands.
'''

import json
from asemi_segmenter.lib import files


#########################################
class CheckpointManager(object):
    '''Checkpoint manager keep track of which stages in a process are complete.'''

    #########################################
    def __init__(self, this_command, checkpoint_fullfname, reset_checkpoint=False, initial_content=None):
        '''
        Create a new checkpoint manager.

        :param str this_command: The unique name of the command using the checkpoint (serves as
            a namespace).
        :param str checkpoint_fullfname: The full file name (with path) of the checkpoint file. If
            None then the checkpoint state is not persisted.
        :param bool reset_checkpoint: Whether to clear the checkpoint from the file (if it
            exists) and start afresh.
        :param dict initial_content: The checkpoint data to initialise the checkpoint with.
            If the checkpoint file already contains data for this checkpoint, then a dictionary
            update will be made such that any keys in the file's checkpoint which are also in the
            initial_content will be overwritten, but nothing else. This is only the checkpoint
            for the command in question, not the entire checkpoint file content. In other words,
            only a single dictionary should be in initial_content, not 2 nested dictionaries.
            If None then the checkpoint will either be empty or the content of the checkpoint file if it exists. Can be used to reset the checkpoint of the command by passing in
            an empty dictionary.
        '''
        self.this_command = this_command
        self.checkpoint_fullfname = checkpoint_fullfname
        self.checkpoints_ready = dict()
        if self.checkpoint_fullfname is not None and files.fexists(self.checkpoint_fullfname):
            with open(self.checkpoint_fullfname, 'r', encoding='utf-8') as f:
                self.checkpoints_ready = json.load(f)
        if this_command not in self.checkpoints_ready or reset_checkpoint:
            self.checkpoints_ready[this_command] = dict()
        if initial_content is not None:
            self.checkpoints_ready[this_command].update(initial_content)
        if self.checkpoint_fullfname is not None:
            with open(self.checkpoint_fullfname, 'w', encoding='utf-8') as f:
                json.dump(self.checkpoints_ready, f, indent='\t')

    #########################################
    def get_next_to_process(self, this_checkpoint):
        '''
        Get next iteration to process according to checkpoint.

        :param str this_checkpoint: The unique name of the current checkpoint.
        :return The next iteration number.
        :rtype int
        '''
        if (
                self.this_command in self.checkpoints_ready and
                this_checkpoint in self.checkpoints_ready[self.this_command]
            ):
            return self.checkpoints_ready[self.this_command][this_checkpoint]
        return 0

    #########################################
    def apply(self, this_checkpoint):
        '''
        Apply a checkpoint for use in a with block.

        If the checkpoint was previously completed then the context manager will return an
        object that can be raised to skip the with block. If not then it will return None
        and at the end of the block will automatically save that the checkpoint was completed.

        Example
        .. code-block:: python
            checkpoint_manager = Checkpoint('command_name', 'checkpoint/fullfname.json')
            with checkpoint_manager.apply('checkpoint_name') as ckpt:
                if ckpt is not None:
                    raise ckpt
                #Do something.
            #Now the checkpoint 'checkpoint_name' has been recorded as completed (or was skipped).

        :param str this_checkpoint: The unique name of the current checkpoint.
        :return A context manager.
        '''

        class SkipCheckpoint(Exception):
            '''Special exception for skipping the checkpoint with block.'''
            pass

        class ContextMgr(object):
            '''Context manager for checkpoints.'''

            def __init__(self, checkpoint_obj):
                self.checkpoint_obj = checkpoint_obj

            def __enter__(self):
                if (
                        self.checkpoint_obj.this_command in \
                        self.checkpoint_obj.checkpoints_ready and
                        this_checkpoint in self.checkpoint_obj.checkpoints_ready[
                            self.checkpoint_obj.this_command
                            ]
                    ):
                    return SkipCheckpoint()
                return None

            def __exit__(self, etype, ex, traceback):
                if etype is SkipCheckpoint:
                    return True
                elif etype is None:
                    if self.checkpoint_obj.checkpoint_fullfname is not None:
                        if (
                                self.checkpoint_obj.this_command not in \
                                self.checkpoint_obj.checkpoints_ready
                            ):
                            self.checkpoint_obj.checkpoints_ready[
                                self.checkpoint_obj.this_command] = dict()
                        if this_checkpoint not in self.checkpoint_obj.checkpoints_ready[
                                self.checkpoint_obj.this_command
                            ]:
                            self.checkpoint_obj.checkpoints_ready[
                                self.checkpoint_obj.this_command
                                ][this_checkpoint] = 0
                        self.checkpoint_obj.checkpoints_ready[
                            self.checkpoint_obj.this_command
                            ][this_checkpoint] += 1
                        with open(self.checkpoint_obj.checkpoint_fullfname, 'w', encoding='utf-8') as f:
                            json.dump(self.checkpoint_obj.checkpoints_ready, f, indent='\t')
                return None
        return ContextMgr(self)