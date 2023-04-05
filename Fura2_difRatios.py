import tkinter as tk
from tkinter import ttk
import pandas as pd

import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.pyplot as plt
from tkinter import filedialog as fd 
from tkinter import messagebox as mb

class coverslip():
    def __init__(self, logfile, exp_params):
        self.log_file = logfile
        self.exp_params = exp_params

        self.__load_FURA_log()
        self.__clean_raw_data()
        self.__get_metrics()
        self.__get_delta_ratio()
        self.__determine_responsive()

    

    def set_params(self, params):
        self.exp_params = params;
        self.__clean_raw_data()
        self.__get_metrics()
        self.__get_delta_ratio()
        self.__determine_responsive()



    def __load_FURA_log(self):

        f = open(self.log_file, "r")
        dat = f.read()
        f.close()

        split_dat = dat.split('\n')
        
        if 'File' in split_dat[0]:
            file_name = split_dat[0].split(': ')[1][:-1]

        if 'Date' in split_dat[1]:
            date = split_dat[1].split(': ')[1][:-1]


        regions = [line for line in split_dat if line.startswith('"Region')]
        number_ROIs = len(regions)


        if number_ROIs == 0:
            print("No header in file")
            num_dat = [sub.split(",") for sub in split_dat[2:]]
            data_table = pd.DataFrame(num_dat, dtype = float)
            self.regions = [];

        else:
            region_table = self.get_region_table(regions)
            column_names = [split_dat[number_ROIs+2].split(', ')][0]
            cn = [n.replace('"', '') for n in column_names]
            roi_id = [n.split(" ",1)[0] for n in cn]
            data_id = [n.split(" ",1)[1] for n in cn]

            num_dat = [sub.split(",") for sub in split_dat[(number_ROIs+3):]]
            data_table = pd.DataFrame(num_dat, columns = cn, dtype = float) 
        
        self.datafile = file_name
        self.date = date
        self.log_data = data_table

    def get_region_table(self, regions):

        col_names = ["Region", "Location X", "Location Y", "Size X", "Size Y", "Area"]
        region_data = [re.findall(r'\d+', ROI) for ROI in regions]
        region_table = pd.DataFrame(region_data, columns= col_names)
        self.regions = region_table

    def __clean_raw_data(self):

        forty = self.log_data.iloc[:,1::3]
        eighty = self.log_data.iloc[:,2::3]

        forty = forty.subtract(forty.iloc[:,0], axis = 0);
        eighty = eighty.subtract(eighty.iloc[:,0], axis = 0)


        #ratio_dat = pd.DataFrame(forty.values/eighty.values)
        ratio_dat = self.log_data.iloc[:,3::3]
        

        ratio_dat.iloc[:,0] = ratio_dat.iloc[:,0].fillna(0)
        
        ratio_dat.columns = [x  for x in range(len(ratio_dat.columns))]
        time = self.log_data.iloc[:,0];
        delta_time = time - time[0];

        
        ratio_dat.insert(0, 'Time', time)
        ratio_dat.insert(1, 'Elapsed', delta_time)


        id_values = [n for (a,n) in zip(self.exp_params['lp_frames'], self.exp_params['ids']) for x in range(a)]

        if len(id_values) < len(ratio_dat):
            added = len(ratio_dat) - len(id_values)
            [id_values.append("None") for x in range(added)]
            
            if added > 10:
                mb.showwarning(message = 'Experiment is '+str(added/10)+ ' minutes longer than parameters indicate')


                

        if len(id_values) > len(ratio_dat):
            raise Exception("Parameters indicate longer experiment than data given")
        

    

    
        ratio_dat.insert(2, 'LP', id_values)
        ratio_dat = ratio_dat.drop(0, axis = 1)

        self.ratio_dat = ratio_dat
        
        

        


    def __get_metrics(self):

        grouped = self.ratio_dat.iloc[:,2:].groupby('LP', sort = False)
        
        mean_val = grouped.mean()
        
        max_val = grouped.max()
        min_val = grouped.min()


        metrics_dat = pd.DataFrame(columns = self.ratio_dat.columns[3:])
        
        for m,n in zip(self.exp_params['metrics'], self.exp_params['ids']):
            
    
            if m == 2:
                metrics_dat.loc[n] = mean_val.loc[n]
                
            
            if m == 1:
                metrics_dat.loc[n] = min_val.loc[n]
            
            if m == 3:
                metrics_dat.loc[n] = max_val.loc[n]
        
        

        self.delta_dat = (metrics_dat- metrics_dat.iloc[0])/metrics_dat.iloc[0]
        self.metrics_dat = metrics_dat
        
    
    def __get_delta_ratio(self): 

        grouped = self.ratio_dat.iloc[:,2:].groupby('LP', sort = False)
        mean_val = grouped.mean()
       
        drr = self.ratio_dat.copy()
        drr = drr.drop('Elapsed', axis = 1)
        drr = drr.drop('LP', axis = 1)
        drr = drr.drop('Time', axis = 1)
        
       
        drr = (drr - mean_val.iloc[0,:])/(mean_val.iloc[0,:])
  
    
        self.drr = drr
    
    def __determine_responsive(self):


        thresh_check = np.array(self.exp_params['tc'])
        tc = np.array(self.exp_params['ids'])[thresh_check > 0]
        tv = np.array(self.exp_params['threshold'])[thresh_check > 0]
        tr = np.array(self.exp_params['ratio'])[thresh_check > 0]
        td = np.array(self.exp_params['direction'])[thresh_check > 0]

        boo_table = pd.DataFrame(columns = self.metrics_dat.columns)
        for c,v,r,d in zip(tc,tv,tr,td):
            
            if r == 1:
                dat = self.metrics_dat.loc[c]
            elif r == -1:
                dat = self.delta_dat.loc[c]
            else:
                dat = self.delta_dat.loc[c]
            
            if d == 1:
                #boo_array.append(dat < v)
                boo_table.loc[c] = dat > v
            
            elif d == -1:
                #boo_array.append(dat > v)
                boo_table.loc[c] = dat < v


        self.responsive = boo_table


    def output_dat_table(self, boo_array):

        md = self.metrics_dat.loc[:, boo_array]
        dd = self.delta_dat.loc[:, boo_array]
        
        

        metrics_table = pd.DataFrame({'Mean': md.mean(axis = 1), 'Median': md.median(axis = 1), 'STDev': md.std(axis = 1), 
                                    'SEM' : md.sem(axis = 1), 'Min' : md.min(axis = 1), 'Max' : md.max(axis = 1)})

        delta_table = pd.DataFrame({'Mean': dd.mean(axis = 1), 'Median': dd.median(axis = 1), 'STDev': dd.std(axis = 1), 
                                'SEM' : dd.sem(axis = 1), 'Min' : dd.min(axis = 1), 'Max' : dd.max(axis = 1)})   


        return metrics_table, delta_table
    

    def save_data_as_excel(self):
        path = fd.askdirectory()
        if len(path)> 0 : 
            new_fn = self.log_file.split('/')[-1]
            new_fn2 = new_fn.split('.')[0]
            full_path = path+ '/' +new_fn2 + '.xlsx'


            boo_array = (self.responsive.sum()/len(self.responsive)) == 1
            mt, dt = self.output_dat_table(boo_array)

            with pd.ExcelWriter(full_path, mode="w", engine="openpyxl") as writer:
                if len(self.regions) > 0:
                    self.regions.to_excel(writer, sheet_name = 'ROI Information')
                self.log_data.to_excel(writer, sheet_name = 'Raw Data')
                self.ratio_dat.to_excel(writer, sheet_name = 'Ratio Data')
                self.drr.to_excel(writer, sheet_name = 'Delta R over R')
                self.metrics_dat.to_excel(writer, sheet_name = 'Data Metrics')
                self.delta_dat.to_excel(writer, sheet_name = 'Delta Metrics')
                mt.to_excel(writer, sheet_name= 'Responsive Metrics')
                dt.to_excel(writer, sheet_name='Responsive Deltas')
                self.responsive.to_excel(writer, sheet_name = 'Responsive')

            
    

    def plot_coverslip(self):
      

        fig, axs = plt.subplots(2)
        fig.set_size_inches(10,10)
        #ax1 = fig.add_subplot()


        frames = np.cumsum(self.exp_params['lp_frames']).copy()
        
        if 0 not in frames:
            frames = np.append(0, frames)
        

        ts =  self.exp_params['ids']
        if 'End' not in ts:
            ts.append('End')

       
        
        x = self.ratio_dat.loc[self.ratio_dat['LP'] != 'None',:]
        x.iloc[:,3:].plot(legend = False, ax = axs[0])
        
        axs[0].set_title('Ratios')
        for a in frames:
            axs[0].axvline(x = a, color = 'k', label = 'axvline - full height', linestyle = ':')

        axs[0].set_xticks(frames)
        axs[0].set_xticklabels(ts)
        axs[0].set_ylabel('340/380')

        #ax = fig.add_subplot()
        y = self.drr.loc[self.ratio_dat['LP'] != 'None',:]
        y.plot(legend = False, ax = axs[1])
        axs[1].set_title('Delta Ratio')
        for a in np.cumsum(self.exp_params['lp_frames']):
            axs[1].axvline(x = a, color = 'k', label = 'axvline - full height', linestyle = ':')

        axs[1].set_xticks(frames)
        axs[1].set_xticklabels(ts)
        axs[1].set_ylabel('∆R/R')
        self.fig = fig

        return fig
    
    
    def save_coverslip_plot(self):
        path = fd.askdirectory()
        
        if len(path)> 0 : 
        
            new_fn = self.log_file.split('/')[-1]
            new_fn2 = new_fn.split('.')[0]
            full_path = path+ '/' +new_fn2 + '.png'


            plt.savefig(full_path)
            

class LP_frame(tk.Frame):
    def __init__(self, parent, num):
        tk.Frame.__init__(self, parent)

        title_label = tk.Label(self, text = 'Liquid Period '+ str(num+1), font=("Arial Bold", 12))
        self.name_var = tk.StringVar()

        name_label = tk.Label(self, text = 'Drug')
        name_entry = tk.Entry(self,textvariable = self.name_var, width = 15)

        self.time_var = tk.IntVar()
        time_label = tk.Label(self, text = 'Duration (mins)')
        time_entry = tk.Entry(self, textvariable = self.time_var, width = 2)

        self.measurement = tk.IntVar()
        measurement_label = tk.Label(self, text = 'Measurement')
        M1 = tk.Radiobutton(self, text="Min", variable=self.measurement, value=1)
        M2 = tk.Radiobutton(self, text="Mean", variable=self.measurement, value=2)
        M3 = tk.Radiobutton(self, text="Max", variable=self.measurement, value=3)


        self.thresh_check = tk.IntVar()
        c1 = tk.Checkbutton(self, text='Threshold',variable= self.thresh_check, onvalue=1, offvalue=0)
        


        self.direction = tk.IntVar()
        D1 = tk.Radiobutton(self, text=">", variable=self.direction, value=1)
        D2 = tk.Radiobutton(self, text="<", variable=self.direction, value=-1)

        self.ratio = tk.IntVar()
        D3 = tk.Radiobutton(self, text="Ratio", variable=self.ratio, value= 1)
        D4 = tk.Radiobutton(self, text="DR/R", variable=self.ratio, value= -1)
        
        self.thresh = tk.DoubleVar()
        thresh_entry = tk.Entry(self, textvariable = self.thresh, width = 4)


        separator = ttk.Separator(self, orient='horizontal')


        title_label.grid(row = 0, column = 0)
        name_label.grid(row=1,column=0, columnspan = 2)
        name_entry.grid(row=1,column=2, columnspan = 3)

        time_label.grid(row=2,column=0, columnspan= 2)
        time_entry.grid(row=2,column=2)

        measurement_label.grid(row = 3, column = 0, sticky = 'E', columnspan= 2)
        M1.grid(row = 3, column= 3,sticky = 'W')
        M2.grid(row =3, column = 4, sticky = 'W')
        M3.grid(row = 3, column = 5, sticky = 'W')

        c1.grid(row = 4, column = 0)
        D3.grid(row = 4, column= 2, sticky = 'W')
        D4.grid(row =4, column = 3, sticky = 'W')
        D1.grid(row = 4, column = 4, sticky = 'W')
        D2.grid(row = 4, column = 5, sticky = 'W')
        thresh_entry.grid(row =4, column = 6,sticky = 'W')

        separator.grid(row =5, column = 0, columnspan = 7, sticky = 'ew')

        

    def get(self):
        data = {'name': self.name_var.get(), 'time': self.time_var.get(), 'measurement': self.measurement.get(),
                'direction': self.direction.get(), 'threshold': self.thresh.get(), 'thresh_check': self.thresh_check.get(), 'ratio': self.ratio.get()}
    
        return data
   

    def set_data(self, data_dict):
        self.name_var.set(data_dict['name'])
        self.time_var.set(data_dict['time'])
        self.measurement.set(data_dict['measurement'])
        self.direction.set(data_dict['direction'])
        self.thresh.set(data_dict['threshold'])
        self.thresh_check.set(data_dict['thresh_check'])
        self.ratio.set(data_dict['ratio'])


class main_window(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        
        self.wm_title("Fura-2 Analysis")

        
        top_frame = tk.Frame(self)
        top_frame.pack(side="top", fill="both", expand=True)


        lps_label = tk.Label(top_frame, text = 'Enter LP #')
        self.lps = tk.IntVar()
        lps_entry = tk.Entry(top_frame ,textvariable = self.lps, width = 2)
        lps_button =tk.Button(top_frame, text="Submit", command = self.populate_LPs)
        load_param_button = tk.Button(top_frame, text = 'Load Parameter File', command = self.load_parameters)
        save_param_button = tk.Button(top_frame, text = 'Save Current Parameters', command = self.save_parameters)

        lps_label.grid(row = 0, column = 0)
        lps_entry.grid(row = 0, column = 1)
        lps_button.grid(row = 0, column = 2)
        load_param_button.grid(row = 0, column = 3)
        save_param_button.grid(row = 0, column = 4)

        top_frame.pack()

        button_frame = tk.Frame(self)

        load_button = tk.Button(button_frame, text = "Load Files", command = self.load_file)
        cal_button = tk.Button(button_frame, text="Calculate Responses",  command = self.get_data)
        save_res_button = tk.Button(button_frame, text = 'Save Responses',  command = self.save_data)
        plot_button = tk.Button(button_frame, text="Plot Data", command = self.plot_data)
        save_plot_button = tk.Button(button_frame, text = 'Save Plots' , command = self.save_plots)


        load_button.pack(side = 'left')
        cal_button.pack(side = 'left')
        save_res_button.pack(side = 'left')
        plot_button.pack(side = 'left')
        save_plot_button.pack(side = 'left')


        button_frame.pack()

        
        #container = ttk.Frame(self)
        canvas = tk.Canvas(self, height = 600, width = 400)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        #container.pack(side="left", fill="both", expand=False)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        
        self.all_frames = [];
        self.file_path = [];



    def populate_LPs(self):

        if len(self.all_frames) != 0:
            for f in self.all_frames:
                f.destroy()
                self.all_frames = []

            
        for i in range(self.lps.get()):
            f = LP_frame(self.scrollable_frame, i)
            f.pack(side = 'top')
            self.all_frames.append(f)
    

    def load_file(self):

        #dialog = tk.Tk()
        
        #dialog.call('wm', 'attributes', '.', '-topmost', True)
        self.file_path = fd.askopenfilenames()
        #dialog.destroy()

    def get_params(self):
        df = pd.DataFrame([a.get() for a in self.all_frames])


        LP_length = list(df['time'])
        sample_rate = 10
        LP_IDs = list(df['name'])
        metrics = list(df['measurement'])
        thresholds = list(df['threshold'])
        direction = list(df['direction'])
        tc = list(df['thresh_check'])
        r = list(df['ratio'])

        lp_frames = [i * sample_rate for i in LP_length]
        exp_params = {"lp_frames":lp_frames, "sample_rate":sample_rate, "ids":LP_IDs, "metrics": metrics, "threshold": thresholds, 
                        "direction": direction, 'ratio': r, 'tc': tc}
        
        return exp_params

    def get_data(self):
        
        if len(self.file_path) == 0:
            mb.showwarning(message = 'No log file imported')
        elif len(self.all_frames) == 0:
            mb.showwarning(message = 'No parameters set')
        else:
            exp_params = self.get_params()
            
            
                
            exp_params = self.get_params()
            self.all_cs = []
            for fn in self.file_path:
                
                cs = coverslip(fn, exp_params)
                self.all_cs.append(cs)

            

                newWindow = tk.Toplevel(self)
                name_title = fn.split('/')[-1]
                newWindow.title(name_title)


                all_resp_frame = tk.Frame(newWindow)
                resp_frame = tk.Frame(all_resp_frame)
                resp_title = tk.Label(resp_frame, text = 'Number of responsive cells', font =("Arial Bold", 12)  )
                resp_title.pack(side = 'top')
                all_counts = cs.responsive.sum(axis = 1)
                resps = [name+': '+str(count) for count, name in zip(all_counts, all_counts.index)]
                for r in resps:
                    label = tk.Label(resp_frame, text = r )
                    label.pack(side = 'top')
                
                tot = (cs.responsive.sum()/len(cs.responsive) == 1).sum()
                label2 = tk.Label(resp_frame, text = 'All: '+str(tot))
                label2.pack(side = 'top')

                resp_frame.pack(side = 'left', anchor = 'nw')

                per_frame = tk.Frame(all_resp_frame)
                per_title = tk.Label(per_frame, text = 'Percent of responsive cells', font =("Arial Bold", 12)  )
                per_title.pack(side = 'top')
                all_per = (cs.responsive.sum(axis = 1)/len(cs.responsive.columns))*100
                pers = [name+': '+str(count)+'%' for count, name in zip(all_per, all_per.index)]
                for r in pers:
                    label = tk.Label(per_frame, text = r )
                    label.pack(side = 'top')
                
                tot2 = (tot/len(cs.responsive.columns)) *100
                label2 = tk.Label(per_frame, text = 'All: '+str(tot2)+ '%')
                label2.pack(side = 'top')

                per_frame.pack(side = 'right', anchor = 'ne')
                all_resp_frame.pack(side = 'top')


                boo_array = (cs.responsive.sum()/len(cs.responsive)) == 1
                mt, dt = cs.output_dat_table(boo_array)


                metrics_title = tk.Label(newWindow, text = 'Summary Stats: Ratios', font=("Arial Bold", 12) )
                metrics_text = tk.Label(newWindow, text = mt.to_string())
                delta_title = tk.Label(newWindow, text = 'Summary Stats: Delta R/R' , font=("Arial Bold", 12))
                delta_text = tk.Label(newWindow, text = dt.to_string())


                metrics_title.pack(side = 'top')
                metrics_text.pack(side = 'top')
                delta_title.pack(side = 'top')
                delta_text.pack(side = 'top')
    
    def save_data(self):
        
        if len(self.file_path) == 0:
            mb.showwarning(message = 'No log file imported')
        elif len(self.all_frames) == 0:
            mb.showwarning(message = 'No parameters set')
        else:
            exp_params = self.get_params()
            if len(self.all_cs) == 0:
                for fn in self.file_path:
                    cs = coverslip(fn, exp_params)
                    self.all_cs.append(cs)


            for cs in self.all_cs:
                cs.save_data_as_excel()
                
                        
    def plot_data(self):

        if len(self.file_path) == 0:
            mb.showwarning(message = 'No log file imported')
        elif len(self.all_frames) == 0:
            mb.showwarning(message = 'No parameters set')
        else:
            
            for cs in self.all_cs:
                f = cs.plot_coverslip()
                newWindow = tk.Toplevel(self)
                newWindow.title(cs.log_file)
                canvas = FigureCanvasTkAgg(f, master = newWindow)  
                canvas.draw()

                #toolbar = NavigationToolbar2Tk(canvas, newWindow, pack_toolbar=False)
                #toolbar.update()
                
                    # placing the canvas on the Tkinter window
                canvas.get_tk_widget().pack()
    

    def save_plots(self):
        for cs in self.all_cs:
            cs.save_coverslip_plot()



    def load_parameters(self):
        #dialog = tk.Tk()
        
        #dialog.call('wm', 'attributes', '.', '-topmost', True)
        self.parameter_file = fd.askopenfilename()
        #dialog.destroy()

        param_data = pd.read_excel(self.parameter_file)

        if len(self.all_frames) != 0:
            for f in self.all_frames:
                f.destroy()
                self.all_frames = []

        for index, row in param_data.iterrows():
            f = LP_frame(self.scrollable_frame, index)
            f.set_data(row.to_dict())
            f.pack(side = 'top')
            self.all_frames.append(f)


    
    def save_parameters(self):


        if len(self.all_frames) == 0:
            mb.showwarning(message = 'No parameters set')
        else:
            params = pd.DataFrame([a.get() for a in self.all_frames])
            
            try:
                # with block automatically closes file
                with fd.asksaveasfile(mode='w', defaultextension=".xlsx") as file:
                    params.to_excel(file.name)
                    file.close()
            except AttributeError:
                # if user cancels save, filedialog returns None rather than a file object, and the 'with' will raise an error
                print("The user cancelled save")



if __name__ == '__main__':
    test1 = main_window()
    test1.mainloop()